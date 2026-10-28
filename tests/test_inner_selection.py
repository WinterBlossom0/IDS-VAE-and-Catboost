import json
import os

import numpy as np
import pandas as pd

from idsvae import config
from idsvae.data import splits
from idsvae.experiments import inner_selection
from idsvae.framework import correction


def test_best_keeps_the_current_value_on_a_tie_and_takes_a_strictly_better_one():
    assert inner_selection.best((0.1, 0.2, 0.5), {0.1: 0.9, 0.2: 0.9, 0.5: 0.8}, current=0.2) == 0.2
    assert inner_selection.best((0.1, 0.2, 0.5), {0.1: 0.95, 0.2: 0.9, 0.5: 0.8}, current=0.2) == 0.1


def test_every_current_rule_setting_is_one_of_its_candidates():
    current = correction.RuleSettings()
    for name, candidates in inner_selection.RULE_CANDIDATES.items():
        assert getattr(current, name) in candidates


class FakeRuleRun:
    validation_predicted = "stage3"

    def predictions(self, rule_settings):
        return rule_settings.cell_width

    def report(self, predicted):
        return {"value": predicted}


def test_parallel_rule_reports_keep_the_task_order():
    settings = correction.RuleSettings()
    tasks = [(key, value) for key in ("a", "b") for value in (None, settings, inner_selection.dataclasses.replace(settings, cell_width=0.5))]
    reports = inner_selection.rule_reports({"a": FakeRuleRun(), "b": FakeRuleRun()}, tasks)
    assert [report["value"] for report in reports] == ["stage3", 0.2, 0.5] * 2


def synthetic_split(rng, day, rows_per_class, offset):
    parts = []
    for label, rows in rows_per_class.items():
        parts.append(pd.DataFrame({"f0": offset + np.arange(rows, dtype="float32"), "label": label, "day": day,
                                   "time": pd.Timestamp(f"2018-{day} 09:00") + pd.to_timedelta(rng.permutation(rows), unit="s")}))
    return pd.concat(parts, ignore_index=True)


def test_inner_split_divides_fit_train_by_the_hold_out_rule_and_links_the_other_splits(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    source = tmp_path / "data" / "processed"
    source.mkdir(parents=True)
    fit_train = pd.concat([synthetic_split(rng, day, {"Benign": 50, "SSH-Bruteforce": 11}, 0) for day in ("02-14", "02-15")], ignore_index=True)
    fit_train.to_parquet(source / "fit_train.parquet", index=False)
    # Feature values from 10,000 up tell the outer hold-out rows apart from the fit_train rows.
    synthetic_split(rng, "02-15", {"Benign": 20}, 10_000).to_parquet(source / "fit_holdout.parquet", index=False)
    for split in inner_selection.LINKED_SPLITS:
        pd.DataFrame({"split": [split]}).to_parquet(source / f"{split}.parquet")
    monkeypatch.setattr(config, "PROCESSED_DIR", source)
    monkeypatch.setattr(inner_selection, "PROCESSED_DIR", tmp_path / "data" / "final" / "inner_selection" / "processed")
    inner_selection.write_inner_splits()

    inner_dir = inner_selection.PROCESSED_DIR
    inner_validation = splits.assign_seen_holdout(fit_train)
    pd.testing.assert_frame_equal(pd.read_parquet(inner_dir / "fit_holdout.parquet"), fit_train[inner_validation].reset_index(drop=True))
    pd.testing.assert_frame_equal(pd.read_parquet(inner_dir / "fit_train.parquet"), fit_train[~inner_validation].reset_index(drop=True))
    for split in ("fit_train", "fit_holdout"):
        assert (pd.read_parquet(inner_dir / f"{split}.parquet")["f0"] < 10_000).all()
    for split in inner_selection.LINKED_SPLITS:
        link = inner_dir / f"{split}.parquet"
        assert not os.path.isabs(os.readlink(link))
        assert pd.read_parquet(link)["split"].tolist() == [split]


def fold_score(rule_settings, left_out):
    """Mean attack F1 over the folds: cell_width 0.1, 0.2, 0.5 -> 0.5, 0.467, 0.533, although no fold alone prefers 0.5."""
    cell_scores = {"SSH-Bruteforce": {0.1: 1.0, 0.2: 0.0, 0.5: 0.6},
                   "DoS attacks-GoldenEye": {0.1: 0.0, 0.2: 0.9, 0.5: 0.6},
                   "DoS attacks-Slowloris": {0.1: 0.5, 0.2: 0.5, 0.5: 0.4}}
    # burst_quantile 0.975 wins only with cell_width 0.5, so it wins only if the burst step uses the cell width chosen before it.
    best_burst = 0.975 if rule_settings.cell_width == 0.5 else 0.995
    return cell_scores[left_out][rule_settings.cell_width] + (0.05 if rule_settings.burst_quantile == best_burst else 0.0)


class FakeFold:
    validation_predicted = None

    def __init__(self, weight, left_out_class, threads):
        self.left_out_class = left_out_class

    def predictions(self, rule_settings):
        return rule_settings

    def report(self, predicted):
        return {"attack_f1": 0.0 if predicted is None else fold_score(predicted, self.left_out_class)}


def test_rule_settings_are_chosen_one_at_a_time_in_order_on_the_fold_mean(tmp_path, monkeypatch):
    monkeypatch.setattr(inner_selection, "ROOT", tmp_path)
    monkeypatch.setattr(inner_selection, "chosen_topology_weight", lambda: 1.0)
    monkeypatch.setattr(inner_selection, "RuleRun", FakeFold)
    monkeypatch.setattr(inner_selection, "raw_report", lambda fold, left_out: {"attack_f1": 0.0})
    inner_selection.choose_rule_settings(threads=1)
    output = json.loads((tmp_path / "rule_choice.json").read_text())
    assert [step["setting"] for step in output["steps"]] == list(inner_selection.RULE_CANDIDATES)
    assert output["steps"][0]["mean_attack_f1"]["0.5"] == np.mean([0.6, 0.6, 0.4])
    defaults = correction.RuleSettings()
    # Settings with equal scores for every candidate keep their start values.
    expected = {**inner_selection.dataclasses.asdict(defaults), "cell_width": 0.5, "burst_quantile": 0.975}
    assert output["chosen"] == expected
