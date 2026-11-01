#!/usr/bin/env bash
# Download an ANN-Benchmarks dataset in HDF5 form (vectors + exact neighbours).
#   bash scripts/download_ann.sh sift-128-euclidean     # SIFT1M, 1M x 128, ~500 MB
#   bash scripts/download_ann.sh gist-960-euclidean     # GIST1M, 1M x 960, ~3.6 GB
#   bash scripts/download_ann.sh glove-100-angular      # GloVe, 1.18M x 100
set -euo pipefail
name="${1:-sift-128-euclidean}"
mkdir -p data/ann
out="data/ann/${name}.hdf5"
if [[ -s "$out" ]]; then echo "already present: $out"; exit 0; fi
curl -fL --retry 3 -o "$out.part" "http://ann-benchmarks.com/${name}.hdf5"
mv "$out.part" "$out"
python - "$out" <<'EOF'
import sys, h5py
with h5py.File(sys.argv[1]) as f:
    print({k: f[k].shape for k in f}, dict(f.attrs))
EOF
