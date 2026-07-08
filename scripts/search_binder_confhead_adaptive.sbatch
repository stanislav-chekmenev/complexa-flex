#!/bin/bash
#SBATCH --job-name=complexa_bon_adaptive
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# --time is the SLURM wallclock. TIME_BUDGET_HOURS (default 16h) is checked only
# BETWEEN rounds, so a round already in flight always finishes: the loop can
# overshoot the budget by up to one round's duration (round 1 = 3000 samples, no
# per-round cutoff). Size --time against worst-case (round-1 length + one extra
# round + AF2 eval), NOT against 16h. 20h leaves ~4h of margin.
#SBATCH --time=20:00:00
#SBATCH --output=logs/bon_adaptive_%j.out
#SBATCH --error=logs/bon_adaptive_%j.err

# Adaptive best-of-N confidence-head binder search. Runs repeated generate ->
# evaluate rounds (round 1 = 3000 samples, later rounds = 1000, distinct seeds,
# round-indexed output dirs) until TARGET_SUCCESSES unique AF2-confirmed
# successes accumulate across all rounds OR the cumulative wall-clock reaches
# TIME_BUDGET_HOURS (default 16h, measured over the WHOLE loop). Then renders the
# multi-round successes-vs-GPU-hours plot.
#
# This is a SEPARATE launch path from scripts/search_binder_confhead.sbatch (the
# single-run pipeline). Staging (venv, trunk/AE/confidence ckpts) is identical.

set -euo pipefail

echo "==========================================="
echo "Adaptive confidence-head best-of-N search started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "==========================================="

PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"

# --- One sbatch job = one target. TARGET_TASK is REQUIRED. ---
TARGET_TASK="${TARGET_TASK:-}"
VALID_TARGETS=(
    02_PDL1 29_BHRF1 04_IFNAR2 18_DerF21 33_TrkA 15_DerF7
    13_BBF14 01_PD1 34_Insulin 31_IL7RA 14_CrSAS6 12_Claudin1
)
if [ -z "$TARGET_TASK" ]; then
    echo "ERROR: TARGET_TASK env var is REQUIRED (one sbatch job = one target)." >&2
    echo "       sbatch --export=ALL,TARGET_TASK=02_PDL1 scripts/search_binder_confhead_adaptive.sbatch" >&2
    echo "       Valid targets: ${VALID_TARGETS[*]}" >&2
    exit 1
fi
is_valid_target=0
for t in "${VALID_TARGETS[@]}"; do
    if [ "$t" == "$TARGET_TASK" ]; then is_valid_target=1; break; fi
done
if [ "$is_valid_target" -ne 1 ]; then
    echo "ERROR: TARGET_TASK='$TARGET_TASK' is not one of the 12 easy Fig-21 targets." >&2
    echo "       Valid targets: ${VALID_TARGETS[*]}" >&2
    exit 1
fi
echo "Target: $TARGET_TASK"

source "$PROJECT_ROOT/env.sh"

DATA_ROOT="/netscratch/$USER/complexa-bon"
mkdir -p "$DATA_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# --- Stage virtual environment to node-local scratch (see search_binder_confhead.sbatch). ---
ENV_TARBALL="${ENV_TARBALL:-/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz}"
ENV_LOCAL="$DATA_ROOT/.venv"
ENV_TARBALL_SRC="${ENV_TARBALL_SRC:-$PROJECT_ROOT/venv.tar.gz}"
if [ ! -f "$ENV_TARBALL" ]; then
    echo "venv tarball missing at $ENV_TARBALL; copying from $ENV_TARBALL_SRC ..."
    if [ ! -f "$ENV_TARBALL_SRC" ]; then
        echo "ERROR: source venv tarball not found at $ENV_TARBALL_SRC" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$ENV_TARBALL")"
    cp -v "$ENV_TARBALL_SRC" "$ENV_TARBALL"
fi
echo "Setting up virtual environment on node-local scratch..."
srun --ntasks=1 bash -c "
    if [ ! -d \"$ENV_LOCAL/bin\" ]; then
        mkdir -p \"$DATA_ROOT\"
        if ! tar xzf \"$ENV_TARBALL\" -C \"$DATA_ROOT\"; then
            echo \"ERROR: Failed to unpack $ENV_TARBALL\" >&2; exit 1
        fi
        echo \"\$(hostname): env unpacked to $ENV_LOCAL\"
    else
        echo \"\$(hostname): env already at $ENV_LOCAL\"
    fi
"
if [ ! -d "$ENV_LOCAL/bin" ]; then
    echo "ERROR: Virtual environment not available at $ENV_LOCAL. Aborting." >&2
    exit 1
fi
export VIRTUAL_ENV="$ENV_LOCAL"
export PATH="$ENV_LOCAL/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

# --- Stage trunk + autoencoder + confidence-head checkpoints ---
CKPT_DIR_SRC="$PROJECT_ROOT/ckpts"
CKPT_DIR_DST="$DATA_ROOT/ckpt"
mkdir -p "$CKPT_DIR_DST"
for ckpt_name in complexa.ckpt complexa_ae.ckpt; do
    src="$CKPT_DIR_SRC/$ckpt_name"; dst="$CKPT_DIR_DST/$ckpt_name"
    if [ ! -f "$src" ]; then echo "ERROR: required checkpoint $src not found." >&2; exit 1; fi
    if [ ! -f "$dst" ]; then cp -v "$src" "$dst"; else echo "$dst already staged."; fi
done
CONF_CKPT_PATH="${CONF_CKPT_PATH:-/netscratch/schekmenev/complexa-paedistill/runs/78768/checkpoints/multi-007-2.9509.ckpt}"
if [ ! -f "$CONF_CKPT_PATH" ]; then
    echo "ERROR: confidence-head checkpoint not found at $CONF_CKPT_PATH" >&2; exit 1
fi
CONF_CKPT_DST="$CKPT_DIR_DST/$(basename "$CONF_CKPT_PATH")"
if [ ! -f "$CONF_CKPT_DST" ]; then cp -v "$CONF_CKPT_PATH" "$CONF_CKPT_DST"; else echo "$CONF_CKPT_DST already staged."; fi
export CONF_CKPT_PATH="$CONF_CKPT_DST"

export DATA_PATH="${DATA_PATH:?DATA_PATH must be set (env.sh/.env)}"
export AF2_DIR="${AF2_DIR:?AF2_DIR must be set for the evaluate-stage ColabDesign refold (env.sh/.env)}"
export CKPT_PATH="$CKPT_DIR_DST"

RUN_NAME="bon_adaptive_${TARGET_TASK}_${SLURM_JOB_ID:-local}"
OUTPUT_ROOT="$PROJECT_ROOT/inference/adaptive/${RUN_NAME}"
mkdir -p "$OUTPUT_ROOT"

unset WANDB_API_KEY
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"

# --- Adaptive-loop parameters ---
PIPELINE_CONFIG="$PROJECT_ROOT/configs/search_binder_confhead_adaptive_pipeline.yaml"
TARGET_SUCCESSES="${TARGET_SUCCESSES:-100}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-16}"
ROUND1_NSAMPLES="${ROUND1_NSAMPLES:-3000}"
ROUND_NSAMPLES="${ROUND_NSAMPLES:-1000}"
BASE_SEED="${BASE_SEED:-5}"

echo ""
echo "Configuration:"
echo "  Output root:      $OUTPUT_ROOT"
echo "  Pipeline config:  $PIPELINE_CONFIG"
echo "  Target successes: $TARGET_SUCCESSES"
echo "  Time budget (h):  $TIME_BUDGET_HOURS  (whole loop, generate+evaluate all rounds)"
echo "  Round 1 nsamples: $ROUND1_NSAMPLES"
echo "  Later nsamples:   $ROUND_NSAMPLES"
echo "  Base seed:        $BASE_SEED"
echo ""

echo "Starting adaptive best-of-N confidence search..."
srun --kill-on-bad-exit /usr/bin/env -u WANDB_API_KEY "$ENV_LOCAL/bin/python" \
    -m proteinfoundation.search.adaptive_bon_loop \
    --config-path "$PIPELINE_CONFIG" \
    --task-name "$TARGET_TASK" \
    --run-name "$RUN_NAME" \
    --output-root "$OUTPUT_ROOT" \
    --round1-nsamples "$ROUND1_NSAMPLES" \
    --round-nsamples "$ROUND_NSAMPLES" \
    --target-successes "$TARGET_SUCCESSES" \
    --time-budget-hours "$TIME_BUDGET_HOURS" \
    --base-seed "$BASE_SEED"

# --- Render the multi-round successes-vs-GPU-hours plot from the summary CSV. ---
SUMMARY_CSV="$OUTPUT_ROOT/adaptive_loop_summary.csv"
PLOT_PNG="$OUTPUT_ROOT/successes_vs_gpu_hours_multiround_${RUN_NAME}.png"
if [ -f "$SUMMARY_CSV" ]; then
    echo "Building multi-round plot from $SUMMARY_CSV ..."
    # Assemble --round-dir / --round-wall-clock-hours pairs from the summary CSV
    # (columns: round_index,round_dir,new_unique_successes,wall_clock_hours,...).
    PLOT_ARGS=()
    while IFS=, read -r ridx rdir new wch rest; do
        [ "$ridx" == "round_index" ] && continue
        [ -z "$rdir" ] && continue
        PLOT_ARGS+=(--round-dir "$rdir" --round-wall-clock-hours "$wch")
    done < "$SUMMARY_CSV"
    "$ENV_LOCAL/bin/python" \
        "$PROJECT_ROOT/script_utils/plot/plot_successes_vs_gpu_hours_multiround.py" \
        "${PLOT_ARGS[@]}" \
        --out-png "$PLOT_PNG" \
        --target-name "$TARGET_TASK" \
        --target-successes "$TARGET_SUCCESSES" \
        || echo "WARNING: multi-round plot failed; summary remains at $SUMMARY_CSV" >&2
    echo "Plot written to $PLOT_PNG"
else
    echo "WARNING: adaptive loop summary not found at $SUMMARY_CSV; skipping plot." >&2
fi

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
