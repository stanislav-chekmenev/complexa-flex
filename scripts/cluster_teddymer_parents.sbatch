#!/bin/bash
#SBATCH --job-name=teddymer_seqclust
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/teddymer_seqclust_%j.out
#SBATCH --error=logs/teddymer_seqclust_%j.err

# Offline 30%-sequence-identity clustering of the Teddymer parent monomers.
# Produces dimers.seqclust30.parquet next to the source blob (the source is
# never mutated). CPU-only job: MMseqs2's CPU path is the default and is plenty
# fast for clustering ~532k parents; no GPU, no NCCL, no checkpoints needed.
#
# Usage:
#   sbatch scripts/cluster_teddymer_parents.sbatch
#
# After it finishes, rebuild the deployable blob with:
#   scripts/rebuild_teddymer_blob_version.sh

set -euo pipefail

echo "==========================================="
echo "Teddymer parent seq-id clustering started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "==========================================="

# --- Project layout ---
PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"
VIEW_ROOT="${TEDDYMER_VIEW_ROOT:-/netscratch/$USER/teddymer_v1_blob}"
TMP_ROOT="${CLUSTER_TMP_ROOT:-/netscratch/$USER/teddymer_cluster_tmp}"
DATA_ROOT="/netscratch/$USER/teddymer-seqclust"

mkdir -p "$PROJECT_ROOT/logs" "$TMP_ROOT" "$DATA_ROOT"

# --- MMseqs2 wrapper (Apptainer module) ---
# Loaded BEFORE env.sh so its `mmseqs` lands on PATH; env.sh then exports a
# bogus MMSEQS_EXEC=$UV_VENV/bin/mmseqs (mmseqs is NOT installed in the venv),
# so we unset it afterwards and let cluster_sequences fall back to the
# module's `mmseqs` on PATH (mmseqs_exec=None -> "mmseqs").
module load mmseqs2/2026.06
echo "mmseqs wrapper: $(command -v mmseqs || echo MISSING)"

# --- Source project env, then neutralise the venv-relative MMSEQS_EXEC ---
source "$PROJECT_ROOT/env.sh"
unset MMSEQS_EXEC
echo "MMSEQS_EXEC unset; using module mmseqs on PATH."

# --- Stage virtual environment to node-local scratch ---
# Same labs-NFS -> /netscratch fallback as the training sbatches. The cpu
# partition has its own /netscratch (same path, different physical store), so
# the cross-partition copy from labs NFS is mandatory if the tarball is absent.
ENV_TARBALL="${ENV_TARBALL:-/netscratch/$USER/venvs/complexa-flex/venv.tar.gz}"
ENV_TARBALL_SRC="${ENV_TARBALL_SRC:-$PROJECT_ROOT/venv.tar.gz}"
ENV_LOCAL="$DATA_ROOT/.venv"

if [ ! -f "$ENV_TARBALL" ]; then
    echo "venv tarball missing at $ENV_TARBALL; copying from $ENV_TARBALL_SRC ..."
    if [ ! -f "$ENV_TARBALL_SRC" ]; then
        echo "ERROR: source venv tarball not found at $ENV_TARBALL_SRC" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$ENV_TARBALL")"
    cp -v "$ENV_TARBALL_SRC" "$ENV_TARBALL"
fi

if [ ! -d "$ENV_LOCAL/bin" ]; then
    echo "Unpacking venv to $ENV_LOCAL ..."
    if ! tar xzf "$ENV_TARBALL" -C "$DATA_ROOT"; then
        echo "ERROR: Failed to unpack $ENV_TARBALL" >&2
        exit 1
    fi
else
    echo "venv already staged at $ENV_LOCAL"
fi
if [ ! -d "$ENV_LOCAL/bin" ]; then
    echo "ERROR: venv not available at $ENV_LOCAL. Aborting." >&2
    exit 1
fi
export VIRTUAL_ENV="$ENV_LOCAL"
export PATH="$ENV_LOCAL/bin:$PATH"

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT:$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

# --- Inputs / output ---
DIMERS="$VIEW_ROOT/dimers.parquet"
LOCATOR="$VIEW_ROOT/locator_rows.parquet"
OUT="$VIEW_ROOT/dimers.seqclust30.parquet"

for f in "$DIMERS" "$LOCATOR"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required input missing: $f" >&2
        exit 1
    fi
done

echo ""
echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  View root:    $VIEW_ROOT"
echo "  Dimers:       $DIMERS"
echo "  Locator:      $LOCATOR"
echo "  Output:       $OUT"
echo "  Tmp:          $TMP_ROOT"
echo "  Workers:      ${SLURM_CPUS_PER_TASK:-32}"
echo ""

cd "$PROJECT_ROOT"

srun --kill-on-bad-exit "$ENV_LOCAL/bin/python" scripts/cluster_teddymer_parents.py \
    --dimers   "$DIMERS" \
    --locator  "$LOCATOR" \
    --afdb-root "$VIEW_ROOT" \
    --out      "$OUT" \
    --tmp      "$TMP_ROOT" \
    --num-workers "${SLURM_CPUS_PER_TASK:-32}"

echo "==========================================="
echo "Clustering completed: $(date)"
echo "Output parquet: $OUT"
echo "Next: scripts/rebuild_teddymer_blob_version.sh"
echo "==========================================="
