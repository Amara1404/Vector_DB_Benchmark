"""Download the Amazon Shopping Queries (ESCI) dataset (Apache-2.0), ~1.2 GB.

Cross-platform replacement for download_esci.sh (no bash/curl required).
"""
import os
import urllib.request

import pyarrow.parquet as pq

BASE = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset"
FILES = [
    "shopping_queries_dataset_examples.parquet",
    "shopping_queries_dataset_products.parquet",
]


def main() -> None:
    os.makedirs("data/esci", exist_ok=True)
    for f in FILES:
        out = os.path.join("data", "esci", f)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"already present: {f}")
            continue
        part = f"{out}.part"
        last_err = None
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(f"{BASE}/{f}", part)
                break
            except OSError as e:
                last_err = e
        else:
            raise RuntimeError(f"failed to download {f}") from last_err
        os.replace(part, out)

    products_path = os.path.join("data", "esci", "shopping_queries_dataset_products.parquet")
    print("products:", pq.ParquetFile(products_path).metadata.num_rows)


if __name__ == "__main__":
    main()
