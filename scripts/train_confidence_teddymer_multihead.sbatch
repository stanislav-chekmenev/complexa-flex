#!/bin/bash
#SBATCH --job-name=complexa_multidistill
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100nvl:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/multidistill_%j.out
#SBATCH --error=logs/multidistill_%j.err

set -euo pipefail

echo "==========================================="
echo "Multi-head (pLDDT + PAE) confidence-head distillation started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "==========================================="

# --- Project layout ---
PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"
NUM_DEVICES=2
NUM_NODES=${SLURM_NNODES:-2}
DEVICE_TYPE=h100nvl

# --- Source project env (uses PROJECT_ROOT) ---
# env.sh sources .env (gitignored) and re-exports active tool paths.
# WANDB_ENTITY is read by configs/logging/wandb.yaml via oc.env: resolution;
# WANDB_API_KEY must NOT be present in .env — ~/.netrc (mode 600, readable
# from compute nodes) is the only credential source. The unconditional
# unset below is defence-in-depth.
source "$PROJECT_ROOT/env.sh"

if [ "$DEVICE_TYPE" == "rtx6000" ] || [ "$DEVICE_TYPE" == "h100nvl" ]; then
    DATA_ROOT="/netscratch/$USER/complexa-multidistill"
else
    echo "ERROR: unknown DEVICE_TYPE=$DEVICE_TYPE" >&2
    exit 1
fi

mkdir -p "$DATA_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# --- Stage virtual environment to node-local scratch ---
# Source: tarball at /netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz
# (rebuild via env/build_uv_env.sh + tar czf, then copy here). `uv run` is
# avoided at runtime because the project ships without a uv.lock and
# pyproject.toml carries indirect torch pins resolved via build_uv_env.sh
# (see env/build_uv_env.sh) — `uv run` re-resolves and breaks the venv.
ENV_TARBALL="${ENV_TARBALL:-/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz}"
ENV_LOCAL="$DATA_ROOT/.venv"

# The h100 partition has its own /netscratch storage (confusingly also
# /netscratch/schekmenev) that does NOT share state with the gpu
# partition. If $ENV_TARBALL is missing on the local /netscratch, copy
# the labs-NFS canonical copy into place. Source must be reachable from
# every partition (labs NFS is).
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

# Teddymer view: the dataset's Hydra defaults now point at the staged
# /netscratch blob at `/netscratch/schekmenev/teddymer_v1_blob/` (single
# `data.blob` + locator + dimers + `data.md5` sidecar). The train-start
# integrity gate (`integrity.enabled=true` in
# `distillation_teddymer_multihead.yaml`) verifies the live files against
# the labs-adjacent snapshot at
# `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5` before any
# GPU work. To run against labs directly, override the env vars
# `TEDDYMER_VIEW_ROOT` / `AFDB_PROTEOMES_ROOT` and set
# `+integrity.enabled=false` on the command line.

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
echo "  Nodes:        ${NUM_NODES}"
echo "  GPUs/node:    ${NUM_DEVICES}x $DEVICE_TYPE"
echo "  World size:   $((NUM_NODES * NUM_DEVICES))"
echo ""

# --- WandB auth ---
# Credential source: ~/.netrc (mode 600 on labs home, readable from
# compute nodes). WANDB_API_KEY in .env or the SLURM env is a footgun
# (placeholder string, partial key from a stale paste, or stale value
# from a previous shell). Strip it unconditionally so wandb falls back
# to ~/.netrc cleanly.
unset WANDB_API_KEY

# --- NCCL tuning ---
# h100 partition has no InfiniBand; inter-node NCCL falls back to TCP/Ethernet.
# NCCL_IB_DISABLE=1 is mandatory here (probing IB on a non-IB fabric stalls).
# NCCL_SOCKET_IFNAME can be pinned (e.g. =eth0) if NCCL guesses the wrong
# interface — surface NCCL_DEBUG=INFO first and pin only if needed.
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# --- CUDA allocator: avoid fragmentation under reentrant ckpt + heterogeneous L ---
# Pair-update layers (src/proteinfoundation/nn/modules/pair_update.py:82,88,105,107,110)
# use reentrant torch.utils.checkpoint; the deep backward recompute fragments the
# caching allocator across the 250-550 residue dimer-crop distribution. OOM at
# step 927/63168 (job 53667, single-head PAE run) with 5.15 GiB
# reserved-but-unallocated was textbook. Multi-head adds a second per-pair
# logits tensor (PAE 64-bin) on top of pLDDT's per-residue 50-bin tensor, so
# the fragmentation pressure is at least as high as the single-head PAE run.
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"

# --- Run training ---
# torch.compile is OFF by default (Teddymer dimers are heterogeneously
# cropped; dynamic-L recompile storm risk — see plan §11 abort criterion).
echo "Starting multi-head (pLDDT + PAE) confidence-head distillation..."
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
EXTRA_OVERRIDES=()
if [ -n "$RESUME_CKPT_PATH" ]; then
    echo "Resuming from ckpt: $RESUME_CKPT_PATH"
    EXTRA_OVERRIDES+=("resume_ckpt_path=$RESUME_CKPT_PATH")
fi

srun --kill-on-bad-exit /usr/bin/env -u WANDB_API_KEY "$ENV_LOCAL/bin/python" -m proteinfoundation.confidence.train_confidence \
    --config-name=confidence/distillation_teddymer_multihead \
    trainer.devices=$NUM_DEVICES \
    trainer.num_nodes=$NUM_NODES \
    trainer.precision=bf16-mixed \
    "${EXTRA_OVERRIDES[@]}"

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
