"""Download an ANN-Benchmarks dataset in HDF5 form (vectors + exact neighbours).

  python scripts/download_ann.py sift-128-euclidean     # SIFT1M, 1M x 128, ~500 MB
  python scripts/download_ann.py gist-960-euclidean     # GIST1M, 1M x 960, ~3.6 GB
  python scripts/download_ann.py glove-100-angular      # GloVe, 1.18M x 100

Cross-platform replacement for download_ann.sh (no bash/curl required).
"""
import os
import sys
import urllib.request

import h5py

# ann-benchmarks.com returns 403 for urllib's default User-Agent string.
_UA = "Mozilla/5.0 (compatible; vdb-bench/1.0)"


def _download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "sift-128-euclidean"
    os.makedirs("data/ann", exist_ok=True)
    out = os.path.join("data", "ann", f"{name}.hdf5")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"already present: {out}")
        return

    url = f"http://ann-benchmarks.com/{name}.hdf5"
    part = f"{out}.part"
    last_err = None
    for attempt in range(3):
        try:
            _download(url, part)
            break
        except OSError as e:
            last_err = e
    else:
        raise RuntimeError(f"failed to download {url}") from last_err
    os.replace(part, out)

    with h5py.File(out) as f:
        print({k: f[k].shape for k in f}, dict(f.attrs))


if __name__ == "__main__":
    main()
