#!/bin/bash
#SBATCH --job-name=complexa_bon_search
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=logs/bon_%j.out
#SBATCH --error=logs/bon_%j.err

set -euo pipefail

echo "==========================================="
echo "Confidence-head best-of-N binder search started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "==========================================="

# --- Project layout ---
PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"

# --- One sbatch job = one target. TARGET_TASK is REQUIRED and must be one of the
#     12 "easy" Fig-21 targets. Validate before any expensive staging work. ---
TARGET_TASK="${TARGET_TASK:-}"
VALID_TARGETS=(
    02_PDL1 29_BHRF1 04_IFNAR2 18_DerF21 33_TrkA 15_DerF7
    13_BBF14 01_PD1 34_Insulin 31_IL7RA 14_CrSAS6 12_Claudin1
)
if [ -z "$TARGET_TASK" ]; then
    echo "ERROR: TARGET_TASK env var is REQUIRED (one sbatch job = one target)." >&2
    echo "       Set it via: sbatch --export=ALL,TARGET_TASK=02_PDL1 scripts/search_binder_confhead.sbatch" >&2
    echo "       Valid targets: ${VALID_TARGETS[*]}" >&2
    exit 1
fi
is_valid_target=0
for t in "${VALID_TARGETS[@]}"; do
    if [ "$t" == "$TARGET_TASK" ]; then
        is_valid_target=1
        break
    fi
done
if [ "$is_valid_target" -ne 1 ]; then
    echo "ERROR: TARGET_TASK='$TARGET_TASK' is not one of the 12 easy Fig-21 targets." >&2
    echo "       Valid targets: ${VALID_TARGETS[*]}" >&2
    exit 1
fi
echo "Target: $TARGET_TASK"

# --- Source project env (uses PROJECT_ROOT) ---
# env.sh sources .env (gitignored) and re-exports active tool paths, including
# AF2_DIR / ESM_DIR / FOLDSEEK_EXEC / MMSEQS_EXEC and DATA_PATH. The evaluate
# stage's ColabDesign refold reads AF2 params from ${oc.env:AF2_DIR}
# (configs/pipeline/binder/binder_generate.yaml:135). WANDB_API_KEY must come
# from ~/.netrc only; the unconditional unset below is defence-in-depth.
source "$PROJECT_ROOT/env.sh"

DATA_ROOT="/netscratch/$USER/complexa-bon"
mkdir -p "$DATA_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# --- Stage virtual environment to node-local scratch ---
# Source: tarball at /netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz
# (rebuild via env/build_uv_env.sh + tar czf, then copy here). `uv run` is
# avoided at runtime because the project ships without a uv.lock and
# pyproject.toml carries indirect torch pins resolved via build_uv_env.sh
# — `uv run` re-resolves and breaks the venv.
ENV_TARBALL="${ENV_TARBALL:-/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz}"
ENV_LOCAL="$DATA_ROOT/.venv"

# Each partition has its own /netscratch storage (all mounted at
# /netscratch/schekmenev) that does NOT share state across partitions. If
# $ENV_TARBALL is missing on the local /netscratch, copy the labs-NFS canonical
# copy into place. Source must be reachable from every partition (labs NFS is).
ENV_TARBALL_SRC="${ENV_TARBALL_SRC:-$PROJECT_ROOT/venv.tar.gz}"
if [ ! -f "$ENV_TARBALL" ]; then
    echo "venv tarball missing at $ENV_TARBALL; copying from $ENV_TARBALL_SRC ..."
    if [ ! -f "$ENV_TARBALL_SRC" ]; then
        echo "ERROR: source venv tarball not found at $ENV_TARBALL_SRC" >&2
        echo "       Build via: bash env/build_uv_env.sh --clean && \\" >&2
        echo "                  tar -C \"\$PROJECT_ROOT\" -czf \"\$ENV_TARBALL_SRC\" .venv" >&2
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
            echo \"ERROR: Failed to unpack $ENV_TARBALL\" >&2
            exit 1
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
echo "Virtual environment set up at $ENV_LOCAL"

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

# --- Stage trunk + autoencoder + confidence-head checkpoints ---
# The search runs the frozen complexa trunk + latent autoencoder to sample, and
# the trained confidence multihead to label provisional successes (no AF2 in the
# search loop). Stage all three to node-local scratch.
CKPT_DIR_SRC="$PROJECT_ROOT/ckpts"
CKPT_DIR_DST="$DATA_ROOT/ckpt"
mkdir -p "$CKPT_DIR_DST"

for ckpt_name in complexa.ckpt complexa_ae.ckpt; do
    src="$CKPT_DIR_SRC/$ckpt_name"
    dst="$CKPT_DIR_DST/$ckpt_name"
    if [ ! -f "$src" ]; then
        echo "ERROR: required checkpoint $src not found." >&2
        exit 1
    fi
    if [ ! -f "$dst" ]; then
        cp -v "$src" "$dst"
    else
        echo "$dst already staged."
    fi
done

# Confidence-head checkpoint: default to the trained multihead run; overridable
# via CONF_CKPT_PATH. Stage to node-local scratch and re-point CONF_CKPT_PATH at
# the staged copy so the search config's ${oc.env:CONF_CKPT_PATH} resolves to a
# fast local read.
CONF_CKPT_PATH="${CONF_CKPT_PATH:-/netscratch/schekmenev/complexa-paedistill/runs/78768/checkpoints/multi-007-2.9509.ckpt}"
if [ ! -f "$CONF_CKPT_PATH" ]; then
    echo "ERROR: confidence-head checkpoint not found at $CONF_CKPT_PATH" >&2
    echo "       Override with: sbatch --export=ALL,TARGET_TASK=...,CONF_CKPT_PATH=/path/to.ckpt ..." >&2
    exit 1
fi
CONF_CKPT_DST="$CKPT_DIR_DST/$(basename "$CONF_CKPT_PATH")"
if [ ! -f "$CONF_CKPT_DST" ]; then
    cp -v "$CONF_CKPT_PATH" "$CONF_CKPT_DST"
else
    echo "$CONF_CKPT_DST already staged."
fi
export CONF_CKPT_PATH="$CONF_CKPT_DST"
echo "Checkpoint staging complete. CONF_CKPT_PATH=$CONF_CKPT_PATH"

# --- Pipeline env consumed by the search + evaluate stages ---
# DATA_PATH / AF2_DIR come from env.sh (.env). CKPT_PATH points the config's
# ${oc.env:CKPT_PATH} at the staged trunk/AE dir. AF2_DIR must resolve for the
# ColabDesign refold in the evaluate stage; if env.sh did not set it, the run
# fails at evaluate — set it explicitly here as a placeholder if needed:
#   export AF2_DIR="${AF2_DIR:-$PROJECT_ROOT/community_models/ckpts/AF2}"
export DATA_PATH="${DATA_PATH:?DATA_PATH must be set (env.sh/.env)}"
export AF2_DIR="${AF2_DIR:?AF2_DIR must be set for the evaluate-stage ColabDesign refold (env.sh/.env)}"
export CKPT_PATH="$CKPT_DIR_DST"

RUN_NAME="bon_${TARGET_TASK}_${SLURM_JOB_ID:-local}"

echo ""
echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  Data root:    $DATA_ROOT"
echo "  CKPTs:        $CKPT_DIR_DST"
echo "  Conf ckpt:    $CONF_CKPT_PATH"
echo "  AF2_DIR:      $AF2_DIR"
echo "  DATA_PATH:    $DATA_PATH"
echo "  Run name:     $RUN_NAME"
echo "  Target:       $TARGET_TASK"
echo ""

# --- WandB auth (see training sbatch rationale) ---
unset WANDB_API_KEY

# --- CUDA allocator: avoid fragmentation across heterogeneous crop sizes ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"

# --- Run the full design pipeline (generate w/ confidence scorer -> filter ->
#     evaluate w/ AF2 -> analyze w/ Foldseek). The confidence_scorer block and
#     the pipeline config are owned by a parallel agent; this sbatch only sets
#     CONF_CKPT_PATH and invokes the pipeline. `complexa` == cli_runner:main;
#     runnable as `python -m proteinfoundation.cli.cli_runner`. time_budget_hours
#     caps the pure search at 16h, leaving headroom for setup + the AF2 final
#     filter within the 20h wallclock. ---
PIPELINE_CONFIG="configs/search_binder_confhead_local_pipeline.yaml"
# TIME_BUDGET_HOURS caps the pure search (default 16h; set small for a smoke test).
# EXTRA_OVERRIDES lets a smoke run shrink the sample count, e.g.
#   EXTRA_OVERRIDES="++generation.dataloader.dataset.nres.nsamples=4 ++generation.dataloader.batch_size=4"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-16}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
echo "Starting best-of-N confidence search pipeline (time_budget_hours=$TIME_BUDGET_HOURS)..."
srun --kill-on-bad-exit /usr/bin/env -u WANDB_API_KEY "$ENV_LOCAL/bin/python" \
    -m proteinfoundation.cli.cli_runner design "$PIPELINE_CONFIG" \
    ++generation.task_name="$TARGET_TASK" \
    ++generation.time_budget_hours="$TIME_BUDGET_HOURS" \
    ++run_name="$RUN_NAME" \
    $EXTRA_OVERRIDES

# --- Locate the produced analyze + sidecar CSVs and render the Fig-21 plot ---
# The analyze binder_results_*.csv lands under ./evaluation_results/ and the
# confidence sidecar confidence_scores_*.csv under ./inference/ (both cwd-relative
# under $PROJECT_ROOT). Filter by run_name and take the most-recently-modified
# match so concurrent target jobs do not cross-contaminate.
echo "Locating output CSVs for $RUN_NAME ..."
ANALYZE_CSV="$(find "$PROJECT_ROOT/evaluation_results" -type f -name "binder_results_*.csv" -path "*${RUN_NAME}*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$ANALYZE_CSV" ]; then
    # Fallback: run_name may not appear in the results_dir path; take the newest overall.
    ANALYZE_CSV="$(find "$PROJECT_ROOT/evaluation_results" -type f -name "binder_results_*.csv" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
SIDECAR_CSV="$(find "$PROJECT_ROOT/inference" -type f -name "confidence_scores_*.csv" -path "*${RUN_NAME}*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$SIDECAR_CSV" ]; then
    SIDECAR_CSV="$(find "$PROJECT_ROOT/inference" -type f -name "confidence_scores_*.csv" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi

if [ -n "$ANALYZE_CSV" ] && [ -n "$SIDECAR_CSV" ]; then
    echo "  analyze CSV: $ANALYZE_CSV"
    echo "  sidecar CSV: $SIDECAR_CSV"
    RUN_OUT_DIR="$(dirname "$ANALYZE_CSV")"
    PLOT_PNG="$RUN_OUT_DIR/successes_vs_gpu_hours_${RUN_NAME}.png"
    "$ENV_LOCAL/bin/python" "$PROJECT_ROOT/script_utils/plot/plot_successes_vs_gpu_hours.py" \
        --analyze-csv "$ANALYZE_CSV" \
        --sidecar-csv "$SIDECAR_CSV" \
        --out-png "$PLOT_PNG" \
        --target-name "$TARGET_TASK" \
        || echo "WARNING: plot script failed; CSVs remain at $ANALYZE_CSV / $SIDECAR_CSV" >&2
    echo "Plot written to $PLOT_PNG"
else
    echo "WARNING: could not locate analyze and/or sidecar CSV; skipping plot." >&2
    echo "         analyze='$ANALYZE_CSV' sidecar='$SIDECAR_CSV'" >&2
fi

# --- Rsync run outputs back to project (idempotent; inference + eval trees are
#     already under $PROJECT_ROOT, so this is a no-op unless the pipeline wrote
#     elsewhere — kept for parity with the training sbatch and future overrides). ---
echo "Syncing run outputs back to project..."
DEST_RUN_DIR="$PROJECT_ROOT/ckpts/bon_runs/${SLURM_JOB_ID:-local}"
mkdir -p "$DEST_RUN_DIR"
if [ -n "${ANALYZE_CSV:-}" ]; then
    rsync -av --no-perms --no-owner --no-group "$(dirname "$ANALYZE_CSV")/" "$DEST_RUN_DIR/" || {
        echo "WARNING: rsync of run dir failed; outputs remain at $(dirname "$ANALYZE_CSV")" >&2
    }
fi

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
