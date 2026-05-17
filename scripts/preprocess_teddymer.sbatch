#!/bin/bash
#SBATCH --job-name=teddymer_preprocess
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/teddymer_preprocess_%j.out
#SBATCH --error=logs/teddymer_preprocess_%j.err

set -euo pipefail

echo "==========================================="
echo "Teddymer preprocessing started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "==========================================="

# --- Project layout ---
PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"

# --- Inputs / outputs (overridable via env) ---
STAGING="${TEDDYMER_STAGING:-/mnt/storage01/home/schekmenev/data/teddymer_v1/_raw}"
INVENTORY="${AFDB_INVENTORY:-/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches}"
PROTEOMES="${AFDB_PROTEOMES:-/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4}"
OUT_DIR="${TEDDYMER_OUT:-/mnt/storage01/home/schekmenev/data/teddymer_v1}"
SANITY_N="${SANITY_N:-50}"
ON_MISSING="${ON_MISSING:-drop}"

mkdir -p "$PROJECT_ROOT/logs" "$OUT_DIR"

cd "$PROJECT_ROOT"

# Use the uv-managed venv at the project root; do not stage to node-local
# scratch — this is a single-node CPU job and the venv is already on shared FS.
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "ERROR: project venv at $PROJECT_ROOT/.venv not found." >&2
    exit 1
fi
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1

echo "Configuration:"
echo "  staging        = $STAGING"
echo "  inventory      = $INVENTORY"
echo "  afdb-proteomes = $PROTEOMES"
echo "  out            = $OUT_DIR"
echo "  on-missing     = $ON_MISSING"
echo "  sanity-n       = $SANITY_N"
echo ""

python -m proteinfoundation.datasets.teddymer.build_view \
    --staging "$STAGING" \
    --inventory "$INVENTORY" \
    --afdb-proteomes "$PROTEOMES" \
    --out "$OUT_DIR" \
    --on-missing "$ON_MISSING" \
    --sanity-n "$SANITY_N"

echo "==========================================="
echo "Teddymer preprocessing done: $(date)"
echo "==========================================="
