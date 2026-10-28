"""Detector tree depth: the held-out selection scores of detector_runs.held_out_report with deeper monotone trees, for the
chosen novelty reference and input set. Depth 2 is the adopted depth; its scores are in selection_multi_{reference}.json.
No hold-out or test-day data enter these scores."""
import json

from idsvae import files
from idsvae.experiments import detector_runs, final_run
from idsvae.framework import detector

DEPTHS = (3, 4)


def depth_scores(depth, feature_set):
    reports = {seed: detector_runs.held_out_report(seed, feature_set, depth) for seed in final_run.SEEDS}
    detection, false_alarms = detector_runs.summary_scores(reports)
    return {"mean_detection": detection, "mean_benign_false_alarms": false_alarms}


def main():
    feature_set = detector_runs.chosen_feature_set_multi()
    selection = json.loads((detector_runs.ROOT / f"selection_multi_{detector_runs.NOVELTY_REFERENCE}.json").read_text())
    output = {"feature_set": feature_set, str(detector.DEPTH): selection["scores"][feature_set]}
    for depth in DEPTHS:
        output[str(depth)] = depth_scores(depth, feature_set)
        print(depth, json.dumps(output[str(depth)]), flush=True)
    files.write_json_atomic(detector_runs.ROOT / f"depths_{detector_runs.NOVELTY_REFERENCE}.json", output)


if __name__ == "__main__":
    main()
