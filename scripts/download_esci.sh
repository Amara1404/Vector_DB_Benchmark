#!/usr/bin/env bash
# Download the Amazon Shopping Queries (ESCI) dataset (Apache-2.0), ~1.2 GB.
set -euo pipefail
mkdir -p data/esci
base="https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset"
for f in shopping_queries_dataset_examples.parquet shopping_queries_dataset_products.parquet; do
  if [[ -s "data/esci/$f" ]]; then echo "already present: $f"; continue; fi
  curl -fL --retry 3 -o "data/esci/$f.part" "$base/$f"
  mv "data/esci/$f.part" "data/esci/$f"
done
python -c "import pyarrow.parquet as pq; print('products:', pq.ParquetFile('data/esci/shopping_queries_dataset_products.parquet').metadata.num_rows)"
