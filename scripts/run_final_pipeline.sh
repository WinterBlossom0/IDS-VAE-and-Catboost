#!/usr/bin/env bash
# All runs of the paper that train a VAE or apply the novel-burst rule, in dependency order:
# 1. inner selection: one VAE per topology weight; the base-arm VAEs (no topological loss) use the GPU while the
#    classifiers of the weights tune; then the choice of the weight (src/idsvae/experiments/inner_selection.py);
# 2. the three leave-one-out folds of the rule selection, then every final-run VAE that uses the chosen weight
#    (src/idsvae/experiments/final_run.py); the rule settings are chosen as soon as the fold classifiers are done;
# 3. the raw-feature classifiers of the ablations, then every rule step, in parallel processes.
# The VAEs train one at a time under the shared GPU lock. Classifiers run in CPU_SLOTS parallel slots of 5 threads.
# The VAE and classifier steps skip finished outputs, so a stopped run continues where it stopped; the rule,
# choose-topology and choose-rule steps always recompute, with the same result. Start it with
# `setsid scripts/run_final_pipeline.sh` and stop it with `kill -- -<group id>`.
# Before: run_data.sh. Afterwards, in this order: run_detector.sh (unseen-attack detector),
# run_baselines.sh (comparison methods), run_timing.sh (CPU timing, idle machine), run_paper.sh (paper numbers).
set -uo pipefail
cd "$(dirname "$0")/.."
export IDSVAE_GPU_LOCK="${IDSVAE_GPU_LOCK:-/tmp/idsvae-gpu.lock}"
LOG_DIR=data/final/logs
STEP_LOG_DIR="$LOG_DIR/steps"
SLOT_DIR="$LOG_DIR/cpu_slots"
CPU_SLOTS="${CPU_SLOTS:-3}"
RULE_PROCESSES="${RULE_PROCESSES:-8}"
mkdir -p "$(dirname "$IDSVAE_GPU_LOCK")" "$STEP_LOG_DIR" "$SLOT_DIR"
# Two instances would write the same outputs at the same time.
exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
  echo "another run_final_pipeline.sh is running"
  exit 1
fi
trap 'kill 0' INT TERM

note() { echo "$(TZ=Asia/Kolkata date +%H:%M) $*" >> "$LOG_DIR/pipeline.log"; }

# run_step MODULE ARGS...: one step of idsvae.experiments.MODULE, with its own log file.
run_step() {
  local module=$1
  shift
  local name
  name=$(printf '%s_' "$module" "$@" | tr -c 'A-Za-z0-9_.-' '_')
  if uv run --frozen python -m "idsvae.experiments.$module" "$@" >> "$STEP_LOG_DIR/$name.log" 2>&1; then
    return 0
  fi
  note "FAILED: $module $*"
  return 1
}
export -f run_step note
export LOG_DIR STEP_LOG_DIR

# in_cpu_slot COMMAND...: waits for a free slot (a lock file), then runs the command while it holds the slot.
in_cpu_slot() {
  local slot fd status
  while true; do
    for slot in $(seq 1 "$CPU_SLOTS"); do
      exec {fd}>"$SLOT_DIR/$slot"
      if flock -n "$fd"; then
        "$@"
        status=$?
        exec {fd}>&-
        return "$status"
      fi
      exec {fd}>&-
    done
    sleep 10
  done
}

# stop MESSAGE: lets the started steps finish, then ends the pipeline with an error.
stop() {
  note "stopped: $1"
  wait
  exit 1
}

folds=("SSH-Bruteforce" "DoS attacks-GoldenEye" "DoS attacks-Slowloris")
note "pipeline started"

# Raw-feature CatBoost on the rule-selection folds, for the comparison in rule_choice.json.
fold_raw_pids=()
for left_out in "${folds[@]}"; do in_cpu_slot run_step inner_selection raw --left-out-class "$left_out" & fold_raw_pids+=($!); done

# Phase 1: topology weight candidates, and the base arm while their classifiers tune.
weight_pids=()
for weight in 0.1 0.3 1.0 3.0; do
  if run_step inner_selection vae --topology-weight "$weight"; then
    in_cpu_slot run_step inner_selection classifier --topology-weight "$weight" & weight_pids+=($!)
  fi
done
classifier_pids=()
for seed in 3 4 5; do
  if run_step final_run vae --arm base --seed "$seed"; then
    in_cpu_slot run_step final_run classifier --arm base --seed "$seed" & classifier_pids+=($!)
  fi
done
for pid in "${weight_pids[@]}"; do wait "$pid"; done
if ! run_step inner_selection choose-topology; then
  stop "no topology weight"
fi
weight=$(uv run --frozen python -c "from idsvae.experiments import inner_selection; print(inner_selection.chosen_topology_weight())")
note "topology weight $weight"

# Phase 2: rule-selection folds, then the final-run VAEs with the chosen weight.
for left_out in "${folds[@]}"; do run_step inner_selection vae --topology-weight "$weight" --left-out-class "$left_out"; done
# A subshell cannot wait for the children of this shell, so the raw fold runs are waited for here.
for pid in "${fold_raw_pids[@]}"; do wait "$pid"; done
(
  pids=()
  for left_out in "${folds[@]}"; do in_cpu_slot run_step inner_selection classifier --topology-weight "$weight" --left-out-class "$left_out" & pids+=($!); done
  for pid in "${pids[@]}"; do wait "$pid"; done
  run_step inner_selection choose-rule
) & choose_rule_pid=$!

# Job lines are eval'ed, so that a class name with a space stays one argument.
vae_jobs=()
for seed in 3 4 5; do vae_jobs+=("--arm topology --seed $seed"); done
for seed in 3 4 5; do vae_jobs+=("--arm gaussian --seed $seed"); done
for k in 10 15 23 30 40 64; do vae_jobs+=("--arm topology --seed 3 --feature-count $k"); done
for seed in 3 4 5; do for left_out in "${folds[@]}"; do vae_jobs+=("--arm topology --seed $seed --left-out-class '$left_out'"); done; done
stage3_jobs=()
for arm in latent_only lightgbm xgboost concat; do for seed in 3 4 5; do stage3_jobs+=("--arm $arm --seed $seed"); done; done
raw_jobs=()
for seed in 3 4 5; do raw_jobs+=("--arm raw --seed $seed"); done
for k in 10 15 23 30 40 64; do raw_jobs+=("--arm raw --seed 3 --feature-count $k"); done
for seed in 3 4 5; do for left_out in "${folds[@]}"; do raw_jobs+=("--arm raw --seed $seed --left-out-class '$left_out'"); done; done

for job in "${vae_jobs[@]}"; do
  if eval run_step final_run vae "$job"; then
    eval in_cpu_slot run_step final_run classifier "$job" & classifier_pids+=($!)
  fi
  # The Stage 3 ablations read the design VAEs; they queue when the last design VAE is done.
  if [[ "$job" == "--arm topology --seed 5" ]]; then
    for stage3_job in "${stage3_jobs[@]}"; do eval in_cpu_slot run_step final_run classifier "$stage3_job" & classifier_pids+=($!); done
  fi
done
note "all VAEs done"

# Phase 3: raw-feature classifiers (finished ones are skipped), then all rule steps.
for job in "${raw_jobs[@]}"; do eval in_cpu_slot run_step final_run classifier "$job" & classifier_pids+=($!); done
for pid in "${classifier_pids[@]}"; do wait "$pid"; done
note "all classifiers done"
if ! wait "$choose_rule_pid"; then
  stop "no rule settings"
fi
note "rule settings chosen"
rule_jobs=()
for arm in topology base gaussian latent_only lightgbm xgboost concat; do for seed in 3 4 5; do rule_jobs+=("--arm $arm --seed $seed"); done; done
rule_jobs+=("${raw_jobs[@]}")
for k in 10 15 23 30 40 64; do rule_jobs+=("--arm topology --seed 3 --feature-count $k"); done
for seed in 3 4 5; do for left_out in "${folds[@]}"; do rule_jobs+=("--arm topology --seed $seed --left-out-class '$left_out'"); done; done
printf '%s\n' "${rule_jobs[@]}" | xargs -P "$RULE_PROCESSES" -d '\n' -I {} bash -c 'eval run_step final_run rule "$1"' _ {}
failed=$(awk '/pipeline started/ { count = 0 } /FAILED/ { count++ } END { print count + 0 }' "$LOG_DIR/pipeline.log")
note "pipeline done (failed steps of this run: $failed)"
