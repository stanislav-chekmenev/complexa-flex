#!/bin/bash
#SBATCH --job-name=complexa_multi
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --nodelist=gpu03
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/multi_%j.out
#SBATCH --error=logs/multi_%j.err

set -euo pipefail

echo "==========================================="
echo "pLDDT + PAE confidence-head distillation started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "==========================================="

# --- Project layout ---
PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"
NUM_DEVICES="${NUM_DEVICES:-4}"
NUM_NODES="${NUM_NODES:-1}"
MAX_STEPS="${MAX_STEPS:-}"
DEVICE_TYPE=rtx6000

# --- SLURM/Hydra coherence guard ---
# `trainer.num_nodes` and `trainer.devices` are forwarded to Lightning's
# DDPStrategy and MUST match what SLURM actually allocated (`SLURM_NNODES`
# and `SLURM_NTASKS_PER_NODE`). A silent mismatch lets Lightning launch
# fewer ranks than SLURM expects (or vice-versa), producing one of:
#   - NCCL hang (rendezvous never completes)
#   - silent half-world (some ranks see batch_size, others see 2x)
# Both look identical to "job ran for a while then died" in slurm logs.
# Validate before any expensive staging work.
if [ -n "${SLURM_NNODES:-}" ] && [ "$SLURM_NNODES" != "$NUM_NODES" ]; then
    echo "ERROR: SLURM_NNODES=$SLURM_NNODES != NUM_NODES=$NUM_NODES" >&2
    echo "       Either rerun with #SBATCH --nodes=$NUM_NODES or set NUM_NODES=$SLURM_NNODES" >&2
    exit 1
fi
if [ -n "${SLURM_NTASKS_PER_NODE:-}" ] && [ "$SLURM_NTASKS_PER_NODE" != "$NUM_DEVICES" ]; then
    echo "ERROR: SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE != NUM_DEVICES=$NUM_DEVICES" >&2
    echo "       Either rerun with #SBATCH --ntasks-per-node=$NUM_DEVICES or set NUM_DEVICES=$SLURM_NTASKS_PER_NODE" >&2
    exit 1
fi

# --- Source project env (uses PROJECT_ROOT) ---
# env.sh sources .env (gitignored) and re-exports active tool paths.
# WANDB_ENTITY is read by configs/logging/wandb.yaml via oc.env: resolution;
# WANDB_API_KEY must NOT be present in .env — ~/.netrc (mode 600, readable
# from compute nodes) is the only credential source. The unconditional
# unset below is defence-in-depth.
source "$PROJECT_ROOT/env.sh"

if [ "$DEVICE_TYPE" == "rtx6000" ] || [ "$DEVICE_TYPE" == "h100nvl" ]; then
    DATA_ROOT="/netscratch/$USER/complexa-paedistill"
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

# Each partition has its own /netscratch storage (confusingly all mounted
# at /netscratch/schekmenev) that does NOT share state across partitions.
# If $ENV_TARBALL is missing on the local /netscratch, copy
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

# Teddymer view: this run trains on the leak-free v2 (`seqclust30`) blob,
# which is now the default for this sbatch. The dataset + integrity
# `view_root` resolve `TEDDYMER_VIEW_ROOT` / `AFDB_PROTEOMES_ROOT` via
# oc.env; we export the v2 blob path here (with `:-` so a caller can still
# override on the CLI, e.g. to point at labs) so a bare `sbatch
# scripts/train_confidence_teddymer_multihead.sbatch` picks up v2 without
# any --export plumbing. The h100 /netscratch store is shared across all
# h100 nodes, so the staged blob is visible from every allocated rank.
# The train-start integrity gate (`integrity.enabled=true` in
# `distillation_teddymer_multihead.yaml`) verifies the live files against
# the labs-adjacent v2 snapshot at
# `/mnt/storage01/home/schekmenev/data/teddymer_v2/data.md5` before any
# GPU work. To run against labs directly, override the env vars
# `TEDDYMER_VIEW_ROOT` / `AFDB_PROTEOMES_ROOT` and set
# `+integrity.enabled=false` on the command line.
# NB: this run is single-node on the gpu partition; the v2 blob was
# relocated to the gpu-store /netscratch after the h100-store corruption.
export TEDDYMER_VIEW_ROOT="${TEDDYMER_VIEW_ROOT:-/netscratch/schekmenev/teddymer_v2_blob}"
export AFDB_PROTEOMES_ROOT="${AFDB_PROTEOMES_ROOT:-/netscratch/schekmenev/teddymer_v2_blob}"

export DATA_PATH="$DATA_ROOT"
export CKPT_DIR="$CKPT_DIR_DST"

# --- Run directory for outputs ---
# Exported so the ModelCheckpoint.dirpath in confidence configs
# (`${oc.env:RUN_DIR,...}/checkpoints`) resolves under srun. Without
# `export`, the var is shell-local and the spawned python sees nothing,
# so checkpoints fall back to Lightning's default cwd-relative path.
export RUN_DIR="$DATA_ROOT/runs/${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"

echo ""
echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  Data root:    $DATA_ROOT"
echo "  CKPTs:        $CKPT_DIR_DST"
echo "  Run dir:      $RUN_DIR"
echo "  GPUs:         ${NUM_DEVICES}x $DEVICE_TYPE"
echo ""

# --- WandB auth ---
# Credential source: ~/.netrc (mode 600 on labs home, readable from
# compute nodes). WANDB_API_KEY in .env or the SLURM env is a footgun
# (placeholder string, partial key from a stale paste, or stale value
# from a previous shell). Strip it unconditionally so wandb falls back
# to ~/.netrc cleanly.
unset WANDB_API_KEY

# --- NCCL tuning ---
# gpu partition has InfiniBand (unlike h100) — do NOT disable IB; NCCL uses
# it for inter-node collectives (moot at 1 node but correct default). P2P is
# enabled: single-node 8x rtx6000 want intra-node P2P (NVLink/PCIe) for the
# DDP allreduce; disabling it here would force staged copies through host RAM.
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# --- CUDA allocator: avoid fragmentation under reentrant ckpt + heterogeneous L ---
# Pair-update layers (src/proteinfoundation/nn/modules/pair_update.py:82,88,105,107,110)
# use reentrant torch.utils.checkpoint; the deep backward recompute fragments the
# caching allocator across the 250-550 residue dimer-crop distribution. OOM at
# step 927/63168 (job 53667) with 5.15 GiB reserved-but-unallocated was textbook.
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"

# --- Run training ---
# torch.compile is OFF by default (Teddymer dimers are heterogeneously
# cropped; dynamic-L recompile storm risk — see plan §11 abort criterion).
echo "Starting pLDDT + PAE confidence-head distillation..."
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
EXTRA_OVERRIDES=()
if [ -n "$RESUME_CKPT_PATH" ]; then
    echo "Resuming from ckpt: $RESUME_CKPT_PATH"
    EXTRA_OVERRIDES+=("resume_ckpt_path=$RESUME_CKPT_PATH")
fi
if [ -n "$MAX_STEPS" ]; then
    EXTRA_OVERRIDES+=("trainer.max_steps=$MAX_STEPS")
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
