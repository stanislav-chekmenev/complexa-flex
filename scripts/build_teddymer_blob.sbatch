#!/bin/bash
#SBATCH --job-name=teddymer_blob_repack
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/teddymer_blob_repack_%j.out
#SBATCH --error=logs/teddymer_blob_repack_%j.err

set -euo pipefail

echo "==========================================="
echo "Teddymer blob repack started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "==========================================="

# --- Project layout ---
PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"

# --- Inputs / outputs (overridable via env) ---
LOCATOR="${TEDDYMER_LOCATOR:-/mnt/storage01/home/schekmenev/data/teddymer_v1/locator_rows.parquet}"
LABS_ROOT="${AFDB_PROTEOMES:-/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4}"
OUT_DIR="${TEDDYMER_BLOB_OUT:-/netscratch/schekmenev/teddymer_v1_blob}"
SNAPSHOT_PATH="${TEDDYMER_SNAPSHOT:-/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5}"

mkdir -p "$PROJECT_ROOT/logs" "$OUT_DIR" "$(dirname "$SNAPSHOT_PATH")"

cd "$PROJECT_ROOT"

# Use the uv-managed venv at the project root; CPU-only job — no node-local
# staging needed.
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
echo "  locator        = $LOCATOR"
echo "  labs-root      = $LABS_ROOT"
echo "  out-dir        = $OUT_DIR"
echo "  snapshot-path  = $SNAPSHOT_PATH"
echo ""

python -m proteinfoundation.datasets.teddymer.build_blob \
    --locator "$LOCATOR" \
    --labs-root "$LABS_ROOT" \
    --out-dir "$OUT_DIR" \
    --snapshot-path "$SNAPSHOT_PATH"

echo "==========================================="
echo "Teddymer blob repack done: $(date)"
echo "==========================================="
