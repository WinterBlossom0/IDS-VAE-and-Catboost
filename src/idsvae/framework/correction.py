"""Novel-burst correction after Stage 3 (named "rule" in the code and the result files): a flow that the VAE finds novel
and that belongs to a burst of similar flows takes the attack class that Stage 3 predicted most often for the earlier flows
of the last window.

- Novel: reconstruction NLL above the novelty_quantile of the Benign flows of the calibration day (no attack labels used).
- Burst: MIDAS-R score (Bhatia et al., AAAI 2020, Definition 1 with the temporal decay of MIDAS-R) of the flow's latent
  cell, above the burst_quantile of the scores of the Benign training flows. The cell key replaces the edge of MIDAS
  because the flow records carry no host addresses. Each capture day is its own stream: the MIDAS tick counts from the
  minute of the first flow of that day, so the scores of different days have the same scale.
- Attribution: the attack class most often predicted for strictly earlier flows within window_seconds, when its count
  reaches the minimum support; otherwise the Stage 3 prediction stays. The minimum support is the smallest count above
  the support_quantile of the same count for the Benign flows of the calibration day, where every attack prediction is
  a false alarm.
Every quantity uses only earlier flows: OnlineRule processes one flow at a time, as a deployed detector does, and the
batch functions run the same per-flow code.
"""
import json
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from idsvae import config
from idsvae.data import splits
from idsvae.framework import vae

# MIDAS defaults: decay factor 0.5 per tick and one-minute ticks.
DECAY = 0.5
SECONDS_PER_MINUTE = 60
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class RuleSettings:
    """Settings of the rule. The defaults are the start values of the selection in inner_selection.py, which chooses each value without the hold-out."""
    # Width of a latent cell in training standard deviations of each latent axis.
    cell_width: float = 0.2
    burst_quantile: float = 0.99
    # Kitsune's longest window (Mirsky et al., NDSS 2018) is one minute.
    window_seconds: int = 60
    support_quantile: float = 0.99
    novelty_quantile: float = 0.975


@dataclass
class RuleThresholds:
    novelty: float
    burst: float
    scale: np.ndarray
    min_support: int
    attack_classes: list
    settings: RuleSettings


def epoch_seconds(times):
    return np.asarray(times).astype("datetime64[s]").astype(np.int64)


def tie_key(frame, latent_dim):
    """Order key of the flows of one second: a hash of the VAE outputs of the flow, the same in every split."""
    return splits.row_hashes(frame, [*vae.latent_columns(latent_dim), "recon_nll"])


def time_order(frame, latent_dim):
    seconds = epoch_seconds(frame["time"].to_numpy())
    return seconds, splits.stream_order(seconds, tie_key(frame, latent_dim))


def cell_divisor(scale, cell_width):
    # A latent axis without variance would divide by 0; such an axis then keeps one cell.
    return np.where(scale > 0, scale, 1.0) * cell_width


def cell_key(latent_row, divisor):
    return np.floor(latent_row / divisor).astype(np.int64).tobytes()


class MidasCounter:
    """MIDAS-R chi-square score of each arriving key in one stream (exact counts in place of count-min sketches)."""

    def __init__(self):
        self.first_minute = None
        self.state = {}

    def score(self, key, minute):
        if self.first_minute is None:
            self.first_minute = minute
        now = minute - self.first_minute + 1
        total, decayed, last = self.state.get(key, (0, 0.0, now))
        decayed = decayed * DECAY ** (now - last) + 1.0
        total += 1
        self.state[key] = (total, decayed, now)
        if now == 1:
            return 0.0
        return (decayed - total / now) ** 2 * now ** 2 / (total * (now - 1))


def midas_scores(keys, minutes):
    """Scores of one stream; keys and minutes must be in time order."""
    counter = MidasCounter()
    return np.array([counter.score(key, minute) for key, minute in zip(keys, minutes)])


def stream_scores(frame, latent_dim, scale, cell_width):
    """Burst scores for rows in any order: each capture day is scored as its own time-ordered stream."""
    latent = frame[vae.latent_columns(latent_dim)].to_numpy(np.float64)
    seconds, order = time_order(frame, latent_dim)
    divisor = cell_divisor(scale, cell_width)
    scores = np.empty(len(seconds))
    days = seconds[order] // SECONDS_PER_DAY
    for day in np.unique(days):
        day_rows = order[days == day]
        keys = [cell_key(latent[row], divisor) for row in day_rows]
        scores[day_rows] = midas_scores(keys, seconds[day_rows] // SECONDS_PER_MINUTE)
    return scores


class AttackContext:
    """Counts of the attack predictions of strictly earlier flows within the attribution window."""

    def __init__(self, attack_classes, window_seconds):
        self.attack_classes = list(attack_classes)
        self.window_seconds = window_seconds
        self.counts = dict.fromkeys(self.attack_classes, 0)
        self.window = deque()
        self.current_second = None
        self.current = []

    def top(self, second):
        """(most frequent attack class, its count) among the flows before this second ('', 0 if none); seconds must not decrease."""
        if second != self.current_second:
            # Flows of the same second do not see each other: they enter the window only when a later second arrives.
            for label in self.current:
                self.window.append((self.current_second, label))
                self.counts[label] += 1
            self.current, self.current_second = [], second
        while self.window and self.window[0][0] < second - self.window_seconds:
            self.counts[self.window.popleft()[1]] -= 1
        best, best_count = "", 0
        # A tie between attack classes goes to the class that comes first in attack_classes (the sorted Stage 3 classes).
        for attack in self.attack_classes:
            if self.counts[attack] > best_count:
                best, best_count = attack, self.counts[attack]
        return best, best_count

    def add(self, predicted):
        if predicted in self.counts:
            self.current.append(predicted)


class OnlineRule:
    """The rule for one flow at a time; flows must arrive in time order."""

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.divisor = cell_divisor(thresholds.scale, thresholds.settings.cell_width)
        self.context = AttackContext(thresholds.attack_classes, thresholds.settings.window_seconds)
        self.day = None
        self.midas = MidasCounter()

    def step(self, second, latent_row, recon_nll, predicted):
        """(final label, overridden) of one flow; second is in epoch seconds, predicted is the Stage 3 class."""
        day = second // SECONDS_PER_DAY
        if day != self.day:
            self.day, self.midas = day, MidasCounter()
        burst_score = self.midas.score(cell_key(latent_row, self.divisor), second // SECONDS_PER_MINUTE)
        attack, count = self.context.top(second)
        self.context.add(predicted)
        is_novel = recon_nll > self.thresholds.novelty
        is_burst = burst_score > self.thresholds.burst
        if is_novel and is_burst and count >= self.thresholds.min_support:
            return attack, True
        return predicted, False


def min_support(calib, calib_predicted, attack_classes, latent_dim, settings):
    """Smallest count above the support_quantile of the top attack count before each Benign flow of the calibration day."""
    context = AttackContext(attack_classes, settings.window_seconds)
    seconds, order = time_order(calib, latent_dim)
    is_benign = (calib["label"] == config.BENIGN).to_numpy()
    benign_counts = []
    for row in order:
        count = context.top(int(seconds[row]))[1]
        context.add(calib_predicted[row])
        if is_benign[row]:
            benign_counts.append(count)
    return int(np.floor(np.quantile(benign_counts, settings.support_quantile))) + 1


def thresholds(train, calib, calib_predicted, attack_classes, latent_dim, settings):
    """Rule thresholds from the training streams and the calibration day (its Benign NLL and Stage 3 predictions)."""
    scale = train[vae.latent_columns(latent_dim)].to_numpy(np.float64).std(axis=0)
    train_scores = stream_scores(train, latent_dim, scale, settings.cell_width)
    # Known limit: the training stream holds about 80 % of each day and class, and the scores grow with the count per tick, so fuller streams score higher; inner_selection.py chooses burst_quantile under the same thinning.
    burst = np.quantile(train_scores[(train["label"] == config.BENIGN).to_numpy()], settings.burst_quantile)
    novelty = np.quantile(calib.loc[calib["label"] == config.BENIGN, "recon_nll"].to_numpy(), settings.novelty_quantile)
    support = min_support(calib, np.asarray(calib_predicted, dtype=object), attack_classes, latent_dim, settings)
    return RuleThresholds(float(novelty), float(burst), scale, support, list(attack_classes), settings)


def apply_rule(frame, predicted, rule_thresholds):
    """(new predictions, override mask); frame has the latent means, recon_nll and time, and may span several days."""
    rule = OnlineRule(rule_thresholds)
    latent_dim = len(rule_thresholds.scale)
    latent = frame[vae.latent_columns(latent_dim)].to_numpy(np.float64)
    recon_nll = frame["recon_nll"].to_numpy()
    seconds, order = time_order(frame, latent_dim)
    predicted = np.asarray(predicted, dtype=object)
    changed = predicted.copy()
    override = np.zeros(len(frame), dtype=bool)
    for row in order:
        changed[row], override[row] = rule.step(int(seconds[row]), latent[row], recon_nll[row], predicted[row])
    return changed, override


# ---------------------------------------------------------------- rule inputs of a run

def latent_dim_of(encoded_dir):
    return json.loads((encoded_dir / "summary.json").read_text())["settings"]["latent_dim"]


def load_split(encoded_dir, run_dir, result, part):
    """(encoded rows of one split, Stage 3 predictions for them); fit_train has no prediction file and gives None."""
    encoded = pd.read_parquet(encoded_dir / f"{part}.parquet")
    if part == "fit_train":
        return encoded, None
    predictions = pd.read_parquet(run_dir / result["prediction_files"][part])
    aligned = (len(predictions) == len(encoded) and (predictions["label"].to_numpy() == encoded["label"].to_numpy()).all()
               and (predictions["time"].to_numpy() == encoded["time"].to_numpy()).all())
    if not aligned:
        raise ValueError(f"{part}: prediction rows of {run_dir} do not match the encoded rows of {encoded_dir}")
    return encoded, predictions["predicted"].to_numpy().astype(object)


def fit_stream(train, holdout, holdout_predicted):
    """(stream, attribution context) of the fit days: training and hold-out flows of a day form one stream.

    The training flows count as Benign in the context: their Stage 3 predictions are in-sample and nearly equal their
    labels, so they would bring the training labels into the attribution. A deployed detector has out-of-sample
    predictions for all earlier flows, so the hold-out attribution sees fewer attack predictions than a deployed one.
    """
    stream = pd.concat([train, holdout], ignore_index=True)
    context = np.concatenate([np.full(len(train), config.BENIGN, dtype=object), holdout_predicted])
    return stream, context


def leave_one_out_stream(train, holdout, holdout_predicted, left_out_predictions, left_out_class):
    """(stream, context, threshold rows) of a leave-one-out run, where no fitted stage saw the left-out class.

    The training rows of the left-out class are test rows: they keep their Stage 3 predictions in the context, and the
    thresholds come from the other training rows only. left_out_predictions has the row index in train ("split_row").
    """
    stream, context = fit_stream(train, holdout, holdout_predicted)
    rows = left_out_predictions["split_row"].to_numpy()
    filled = stream.iloc[rows]
    # Every left-out training row needs its prediction once; a missing row would keep the Benign context.
    covers_every_row = len(np.unique(rows)) == len(rows) == int((train["label"] == left_out_class).sum())
    aligned = (covers_every_row and (left_out_predictions["label"].to_numpy() == left_out_class).all()
               and (filled["label"].to_numpy() == left_out_class).all() and (filled["time"].to_numpy() == left_out_predictions["time"].to_numpy()).all())
    if not aligned:
        raise ValueError(f"left-out predictions do not match the {left_out_class!r} training rows at their split_row positions")
    context[rows] = left_out_predictions["predicted"].to_numpy().astype(object)
    return stream, context, train[(train["label"] != left_out_class).to_numpy()]
