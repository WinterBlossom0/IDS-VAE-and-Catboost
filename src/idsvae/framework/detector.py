"""The detector of unseen attack types: a monotone CatBoost on five per-flow features from the VAE outputs and the flow
times, applied to the flows that Stage 3 and the correction call Benign.

Every feature is a quantile on a Benign reference, so the features of different VAEs have one scale. Monotone constraints
(Incer et al., IWSPA 2018: monotone classifiers for detectors that must generalize) let a larger novelty or burst value only
raise the attack score, so the model cannot learn the value ranges of one attack family. The training rows, the cut-off
and the feature references come from experiments/detector_runs.py.
"""
from collections import deque

import numpy as np
from catboost import CatBoostClassifier

from idsvae import config
from idsvae.framework import correction, vae

# Background of a day: the median NLL of its earlier flows outside bursts, from a reservoir of this size, refreshed after
# this many added flows; before the warm-up count the median of the calibration-day Benign flows stands in.
BACKGROUND_RESERVOIR = 20_000
BACKGROUND_REFRESH = 500
BACKGROUND_WARM_UP = 1_000
# Candidate feature sets; each feature has a monotone constraint (+1: a larger value means more likely an attack).
FEATURE_SETS = {
    "novelty_burst": ("nll_q", "burst_q"),
    "novelty_burst_novel": ("nll_q", "burst_q", "novel_burst_q"),
    "novelty_burst_share": ("nll_q", "burst_q", "novel_share_q"),
    "novelty_burst_novel_share": ("nll_q", "burst_q", "novel_burst_q", "novel_share_q"),
    "all": ("nll_q", "kl_tail_q", "burst_q", "novel_burst_q", "novel_share_q"),
}
STREAM_FEATURES = ("burst_q", "novel_burst_q", "novel_share_q")
DEPTH = 2
# A CatBoost model changes with its thread count, so each step keeps the thread count of the reported runs.
THREADS = 8
ITERATIONS = 500
LEARNING_RATE = 0.1
FALSE_ALARM_TARGET = 0.01
UNKNOWN = config.UNKNOWN_ATTACK


def quantile_of(reference, values):
    """Share of the reference values at or below each value."""
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(reference, values, side="right") / len(reference)


class DayBackground:
    """Background NLL of one day: the running median NLL of the earlier flows outside bursts, from a reservoir sample.

    A day can have a different normal NLL level than the calibration day (02-28 Benign has a higher median than 02-23). An
    attack flood is a burst, so its flows do not enter the background and cannot hide themselves.
    """

    def __init__(self, burst_cutoff, fallback_median, rng):
        self.burst_cutoff, self.fallback_median, self.rng = burst_cutoff, fallback_median, rng
        self.new_day()

    def new_day(self):
        self.reservoir, self.seen, self.added_since_refresh, self.median = [], 0, 0, self.fallback_median

    def relative(self, nll):
        return nll - self.median

    def add(self, nll, burst_score):
        if burst_score > self.burst_cutoff:
            return
        self.seen += 1
        if len(self.reservoir) < BACKGROUND_RESERVOIR:
            self.reservoir.append(nll)
        else:
            slot = self.rng.integers(self.seen)
            if slot < BACKGROUND_RESERVOIR:
                self.reservoir[slot] = nll
        self.added_since_refresh += 1
        if self.seen >= BACKGROUND_WARM_UP and self.added_since_refresh >= BACKGROUND_REFRESH:
            self.median, self.added_since_refresh = float(np.median(self.reservoir)), 0


class CellStreams:
    """Per-day stream state: MIDAS-R over all flows and over novel flows, and the novel share of each cell's window."""

    def __init__(self, window_seconds):
        self.window_seconds = window_seconds

    def new_day(self, minute):
        self.all_flows, self.novel_flows, self.windows = correction.MidasCounter(), correction.MidasCounter(), {}
        # Both MIDAS clocks start at the first flow of the day, so the two scores have the same tick scale.
        self.all_flows.first_minute = self.novel_flows.first_minute = minute

    def step(self, key, second, minute, is_novel):
        """(burst score, novel-burst score, novel share of the earlier flows of the cell) of one flow."""
        burst = self.all_flows.score(key, minute)
        novel_burst = self.novel_flows.score(key, minute) if is_novel else 0.0
        window = self.windows.setdefault(key, CellWindow())
        share = window.earlier_novel_share(second, self.window_seconds)
        window.add(second, bool(is_novel))
        return burst, novel_burst, share


def relative_nll(frame, latent_dim, scale, settings, burst_cutoff, fallback_median):
    """NLL minus the background of the same day (DayBackground), for rows in any order."""
    latent = frame[vae.latent_columns(latent_dim)].to_numpy(np.float64)
    nll = frame["recon_nll"].to_numpy(np.float64)
    seconds, order = correction.time_order(frame, latent_dim)
    divisor = correction.cell_divisor(scale, settings.cell_width)
    relative = np.zeros(len(frame))
    background = DayBackground(burst_cutoff, fallback_median, np.random.default_rng(0))
    day = None
    for row in order:
        second = int(seconds[row])
        if second // correction.SECONDS_PER_DAY != day:
            day = second // correction.SECONDS_PER_DAY
            background.new_day()
            counter = correction.MidasCounter()
        relative[row] = background.relative(nll[row])
        background.add(nll[row], counter.score(correction.cell_key(latent[row], divisor), second // correction.SECONDS_PER_MINUTE))
    return relative


def stream_values(frame, latent_dim, scale, is_novel, settings):
    """Per flow, in time order and one stream per day (CellStreams): MIDAS-R score of the latent cell over all flows and
    over novel flows, and the share of novel flows among the strictly earlier flows of the same cell in the window."""
    latent = frame[vae.latent_columns(latent_dim)].to_numpy(np.float64)
    seconds, order = correction.time_order(frame, latent_dim)
    divisor = correction.cell_divisor(scale, settings.cell_width)
    values = np.zeros((3, len(frame)))
    streams = CellStreams(settings.window_seconds)
    day = None
    for row in order:
        second = int(seconds[row])
        minute = second // correction.SECONDS_PER_MINUTE
        if second // correction.SECONDS_PER_DAY != day:
            day = second // correction.SECONDS_PER_DAY
            streams.new_day(minute)
        values[:, row] = streams.step(correction.cell_key(latent[row], divisor), second, minute, is_novel[row])
    return values[0], values[1], values[2]


class CellWindow:
    """Flows of one latent cell in the attribution window, with running counts (a burst can put 10^5 flows in one cell)."""

    def __init__(self):
        self.entries = deque()
        self.count, self.novel = 0, 0
        self.last_second, self.last_count, self.last_novel = None, 0, 0

    def earlier_novel_share(self, second, window_seconds):
        """Share of novel flows among the flows of earlier seconds in the window; flows of the same second do not see each other."""
        while self.entries and self.entries[0][0] < second - window_seconds:
            self.novel -= self.entries.popleft()[1]
            self.count -= 1
        count, novel = self.count, self.novel
        if self.last_second == second:
            count, novel = count - self.last_count, novel - self.last_novel
        return novel / count if count > 0 else 0.0

    def add(self, second, is_novel):
        if self.last_second != second:
            self.last_second, self.last_count, self.last_novel = second, 0, 0
        self.entries.append((second, int(is_novel)))
        self.count += 1
        self.novel += int(is_novel)
        self.last_count += 1
        self.last_novel += int(is_novel)


class OnlineDetector:
    """The detector for one flow at a time, as a deployed detector runs it: the same DayBackground and CellStreams state as the
    batch functions, and the same quantile scales; flows must arrive in time order (one stream, rng seeded per stream)."""

    def __init__(self, columns, scale, settings, novelty, nll_reference, kl_reference, stream_references, background=None):
        self.columns = list(columns)
        self.divisor = correction.cell_divisor(scale, settings.cell_width)
        self.novelty = novelty
        # float64 as in quantile_of; a float32 reference makes searchsorted convert the whole array on every call.
        self.nll_reference, self.kl_reference = (np.sort(np.asarray(reference, dtype=np.float64)) for reference in (nll_reference, kl_reference))
        self.stream_references = [np.sort(np.asarray(reference, dtype=np.float64)) for reference in stream_references]
        self.background = background
        self.streams = CellStreams(settings.window_seconds)
        self.day = None

    def features(self, second, latent_row, nll, kl):
        """Feature values of one flow, in the order of self.columns; second is in epoch seconds."""
        minute = second // correction.SECONDS_PER_MINUTE
        if second // correction.SECONDS_PER_DAY != self.day:
            self.day = second // correction.SECONDS_PER_DAY
            self.streams.new_day(minute)
            if self.background is not None:
                self.background.new_day()
        novelty_value = self.background.relative(nll) if self.background is not None else nll
        stream = self.streams.step(correction.cell_key(latent_row, self.divisor), second, minute, novelty_value > self.novelty)
        if self.background is not None:
            self.background.add(nll, stream[0])
        kl_quantile = scalar_quantile(self.kl_reference, kl)
        values = {"nll_q": scalar_quantile(self.nll_reference, novelty_value), "kl_tail_q": 1 - 2 * min(kl_quantile, 1 - kl_quantile)}
        values.update({name: scalar_quantile(reference, value) for name, reference, value in zip(STREAM_FEATURES, self.stream_references, stream)})
        return [values[column] for column in self.columns]


def scalar_quantile(sorted_reference, value):
    return np.searchsorted(sorted_reference, value, side="right") / len(sorted_reference)


def flags(model, frame, feature_set, threshold):
    return model.predict_proba(frame[list(FEATURE_SETS[feature_set])])[:, 1] > threshold


def combine(stage3_labels, detector_flags):
    """Final label: the Stage 3 label after the correction, or UNKNOWN for a flow that it calls Benign and the detector flags."""
    labels = np.asarray(stage3_labels, dtype=object).copy()
    labels[(labels == config.BENIGN) & detector_flags] = UNKNOWN
    return labels


def fit_detector(features, targets, weights, seed, threads=THREADS, depth=DEPTH):
    model = CatBoostClassifier(iterations=ITERATIONS, depth=depth, learning_rate=LEARNING_RATE, random_seed=seed, verbose=0,
                               thread_count=threads, allow_writing_files=False, monotone_constraints=[1] * features.shape[1])
    model.fit(features, targets, sample_weight=weights)
    return model


def weighted_quantile(values, weights, quantile):
    """Smallest value whose weighted share of values at or below it reaches the quantile."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][min(np.searchsorted(cumulative, quantile), len(values) - 1)])
