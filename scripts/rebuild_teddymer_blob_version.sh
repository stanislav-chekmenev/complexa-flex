#!/bin/bash
# Rebuild a deployable Teddymer view that carries the `seqclust30` cluster
# column, so the train-time integrity gate (datasets/teddymer/integrity.py)
# accepts it. The gate hashes the file literally named `dimers.parquet` at
# view_root against a `data.md5` snapshot AND compares `view_config.yaml:version`
# between the live view and the snapshot — so you cannot just drop the
# clustered parquet beside the old one; the md5 + version must be regenerated.
#
# This script, given the v1 view dir and the clustered parquet produced by
# scripts/cluster_teddymer_parents.py, creates a NEW view dir (v2) and a
# matching snapshot dir, never touching the v1 blob:
#
#   <v2_view>/                          (live, on /netscratch)
#     data.blob              (copied from v1, unchanged)
#     locator_rows.parquet   (copied from v1, unchanged)
#     dimers.parquet         (= the seqclust30 parquet, renamed)
#     data.md5               (regenerated over the three files)
#     view_config.yaml       (version bumped, md5s block refreshed)
#   <v2_snapshot>/data.md5 + view_config.yaml   (copies of the above two)
#
# Then it prints the exact launch overrides.
#
# Usage:
#   scripts/rebuild_teddymer_blob_version.sh \
#     [--v1 /netscratch/$USER/teddymer_v1_blob] \
#     [--clustered <v1>/dimers.seqclust30.parquet] \
#     [--v2 /netscratch/$USER/teddymer_v2_blob] \
#     [--snapshot $HOME/data/teddymer_v2]

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
V1="/netscratch/$USER_NAME/teddymer_v1_blob"
CLUSTERED=""
V2="/netscratch/$USER_NAME/teddymer_v2_blob"
SNAPSHOT="$HOME/data/teddymer_v2"

while [ $# -gt 0 ]; do
    case "$1" in
        --v1)        V1="$2"; shift 2 ;;
        --clustered) CLUSTERED="$2"; shift 2 ;;
        --v2)        V2="$2"; shift 2 ;;
        --snapshot)  SNAPSHOT="$2"; shift 2 ;;
        -h|--help)   sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown arg $1" >&2; exit 1 ;;
    esac
done

CLUSTERED="${CLUSTERED:-$V1/dimers.seqclust30.parquet}"

echo "Rebuilding Teddymer v2 view with seqclust30:"
echo "  v1 source:  $V1"
echo "  clustered:  $CLUSTERED"
echo "  v2 view:    $V2"
echo "  snapshot:   $SNAPSHOT"

for f in "$V1/data.blob" "$V1/locator_rows.parquet" "$V1/view_config.yaml" "$CLUSTERED"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required input missing: $f" >&2
        exit 1
    fi
done

# Verify the clustered parquet actually carries seqclust30 before deploying.
python - "$CLUSTERED" <<'PY'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1], columns=None)
if "seqclust30" not in df.columns:
    sys.exit(f"ERROR: {sys.argv[1]} has no `seqclust30` column; columns={list(df.columns)}")
n_null = int(df["seqclust30"].isna().sum())
if n_null:
    sys.exit(f"ERROR: {sys.argv[1]} has {n_null} null seqclust30 values")
print(f"  seqclust30 OK: {len(df)} dimers, {df['seqclust30'].nunique()} clusters, 0 nulls")
PY

if [ -e "$V2" ]; then
    echo "ERROR: v2 view dir already exists: $V2 (refusing to overwrite)" >&2
    exit 1
fi

mkdir -p "$V2" "$SNAPSHOT"

echo "Copying data.blob (large; may take a minute) ..."
cp -v "$V1/data.blob"            "$V2/data.blob"
cp -v "$V1/locator_rows.parquet" "$V2/locator_rows.parquet"
cp -v "$CLUSTERED"               "$V2/dimers.parquet"

echo "Regenerating data.md5 over the three files ..."
( cd "$V2" && md5sum data.blob dimers.parquet locator_rows.parquet > data.md5 )
cat "$V2/data.md5"

# Mint a v2 version string and refresh view_config.yaml from the v1 template.
# Format mirrors v1: teddymer_v2_blob_<YYYYMMDD>_<dimers_md5_prefix>.
DIMERS_MD5="$(awk '$2=="dimers.parquet"{print $1}' "$V2/data.md5")"
DATE_TAG="$(date -u +%Y%m%d)"
NEW_VERSION="teddymer_v2_blob_${DATE_TAG}_${DIMERS_MD5:0:8}"
echo "New version: $NEW_VERSION"

python - "$V1/view_config.yaml" "$V2" "$NEW_VERSION" <<'PY'
import sys, hashlib, yaml
from pathlib import Path

v1_cfg_path, v2_dir, new_version = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(Path(v1_cfg_path).read_text())

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = ["data.blob", "dimers.parquet", "locator_rows.parquet"]
cfg["version"] = new_version
cfg["md5s"] = {name: md5(v2_dir / name) for name in files}
for key in ("blob_path", "dimers_path", "locator_path"):
    if key in cfg:
        name = {"blob_path": "data.blob", "dimers_path": "dimers.parquet",
                "locator_path": "locator_rows.parquet"}[key]
        cfg[key] = str(v2_dir / name)

(v2_dir / "view_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
print(f"  wrote {v2_dir / 'view_config.yaml'} (version={new_version})")

# Cross-check the data.md5 sidecar agrees with the yaml md5s block.
sidecar = {}
for ln in (v2_dir / "data.md5").read_text().splitlines():
    if ln.strip():
        digest, _, name = ln.partition("  ")
        sidecar[name] = digest
assert sidecar == cfg["md5s"], f"sidecar/yaml md5 mismatch: {sidecar} != {cfg['md5s']}"
print("  data.md5 sidecar and view_config.yaml md5s agree")
PY

echo "Writing snapshot (data.md5 + view_config.yaml) to $SNAPSHOT ..."
cp -v "$V2/data.md5"          "$SNAPSHOT/data.md5"
cp -v "$V2/view_config.yaml"  "$SNAPSHOT/view_config.yaml"

echo ""
echo "==========================================="
echo "v2 view built: $V2"
echo "snapshot:      $SNAPSHOT"
echo ""
echo "Launch training against it with:"
echo "  export TEDDYMER_VIEW_ROOT=$V2"
echo "  export AFDB_PROTEOMES_ROOT=$V2"
echo "  sbatch --export=ALL,TEDDYMER_VIEW_ROOT,AFDB_PROTEOMES_ROOT \\"
echo "    scripts/train_confidence_teddymer_multihead.sbatch \\"
echo "    integrity.snapshot_path=$SNAPSHOT/data.md5"
echo ""
echo "NB: this /netscratch dir is partition-local. Replicate \$V2 to the"
echo "    h100 partition's /netscratch (via labs NFS) before launching there."
echo "    Confirm the run header logs: CLUSTER-AWARE on 'seqclust30' ACTIVE."
echo "==========================================="
