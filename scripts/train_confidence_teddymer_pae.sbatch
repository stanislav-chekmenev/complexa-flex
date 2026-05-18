#!/bin/bash
#SBATCH --job-name=complexa_paedistill
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/paedistill_%j.out
#SBATCH --error=logs/paedistill_%j.err

set -euo pipefail

echo "==========================================="
echo "PAE confidence-head distillation started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "==========================================="

# --- Project layout ---
PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"
NUM_DEVICES=2
DEVICE_TYPE=h100nvl

if [ "$DEVICE_TYPE" == "rtx6000" ] || [ "$DEVICE_TYPE" == "h100nvl" ]; then
    DATA_ROOT="/netscratch/$USER/complexa-paedistill"
else
    echo "ERROR: unknown DEVICE_TYPE=$DEVICE_TYPE" >&2
    exit 1
fi

mkdir -p "$DATA_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# --- Stage virtual environment to node-local scratch ---
STAGE_VENV=${STAGE_VENV:-1}
ENV_TARBALL="$PROJECT_ROOT/venv.tar.gz"
ENV_LOCAL="$DATA_ROOT/.venv"

if [ "$STAGE_VENV" -eq 1 ] && [ -f "$ENV_TARBALL" ]; then
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
else
    echo "Skipping venv staging (STAGE_VENV=$STAGE_VENV); using uv-managed env from project."
fi

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

# --- Stage trunk + autoencoder checkpoints ---
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
echo "Checkpoint staging complete."

# Teddymer view + AFDB parent tars are read from their shared paths via
# the dataset's env defaults (`TEDDYMER_VIEW_ROOT`, `AFDB_PROTEOMES_ROOT`).

export DATA_PATH="$DATA_ROOT"
export CKPT_DIR="$CKPT_DIR_DST"

# --- Run directory for outputs ---
RUN_DIR="$DATA_ROOT/runs/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"

echo ""
echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  Data root:    $DATA_ROOT"
echo "  CKPTs:        $CKPT_DIR_DST"
echo "  Run dir:      $RUN_DIR"
echo "  GPUs:         ${NUM_DEVICES}x $DEVICE_TYPE"
echo ""

# --- NCCL tuning ---
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

cd "$PROJECT_ROOT"

# --- Run training ---
# torch.compile is OFF by default (Teddymer dimers are heterogeneously
# cropped; dynamic-L recompile storm risk — see plan §11 abort criterion).
echo "Starting PAE confidence-head distillation..."
srun --kill-on-bad-exit uv run python -m proteinfoundation.confidence.train_confidence \
    --config-name=confidence/distillation_teddymer_pae \
    trainer.devices=$NUM_DEVICES \
    trainer.strategy=ddp_find_unused_parameters_false \
    trainer.precision=bf16-mixed

# --- Rsync run dir back to project ---
echo "Syncing run dir back to project..."
DEST_RUN_DIR="$PROJECT_ROOT/ckpts/runs/${SLURM_JOB_ID:-local}"
mkdir -p "$DEST_RUN_DIR"
rsync -av --no-perms --no-owner --no-group "$RUN_DIR/" "$DEST_RUN_DIR/" || {
    echo "WARNING: rsync of run dir failed; outputs remain at $RUN_DIR" >&2
}

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
