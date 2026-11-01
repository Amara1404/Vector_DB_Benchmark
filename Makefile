# vdb-bench pipeline. Run `make help`.
PY ?= python
EMBEDDER ?= st:intfloat/multilingual-e5-small
DEVICE ?=

.PHONY: help up up-milvus down install smoke esci-smoke data-sift data-esci esci-embed sift1m esci esci-eval notebook all clean-results

help:
	@echo "make up            start Postgres+pgvector and Qdrant (Docker)"
	@echo "make up-milvus     also start Milvus standalone"
	@echo "make smoke         2-minute check of every adapter on synthetic data"
	@echo "make sift1m        systems benchmark on SIFT1M (downloads ~500 MB)"
	@echo "make esci          ESCI: download, embed 1.8M products, benchmark, fairness eval"
	@echo "make notebook      execute notebooks/results.ipynb -> results/results.html"
	@echo "make all           sift1m + esci + notebook"

up:
	docker compose up -d postgres qdrant
up-milvus:
	docker compose --profile milvus up -d
down:
	docker compose --profile milvus down

install:
	$(PY) -m pip install -r requirements.txt

smoke:
	$(PY) -m bench.runner --config configs/smoke.yaml

data-sift:
	$(PY) scripts/download_ann.py sift-128-euclidean
data-esci:
	$(PY) scripts/download_esci.py

sift1m: data-sift
	$(PY) -m bench.runner --config configs/sift1m.yaml $(if $(ENGINES),--engines $(ENGINES))

data/esci_emb/products.npy: data/esci/shopping_queries_dataset_products.parquet
	$(PY) -m bench.esci_prep --src data/esci --out data/esci_emb --embedder $(EMBEDDER) $(if $(DEVICE),--device $(DEVICE))
data/esci/shopping_queries_dataset_products.parquet:
	$(PY) scripts/download_esci.py
esci-embed: data/esci_emb/products.npy

esci: esci-embed
	$(PY) -m bench.runner --config configs/esci.yaml $(if $(ENGINES),--engines $(ENGINES))
	$(PY) -m bench.esci_eval --run results/esci --emb data/esci_emb

esci-smoke: data/esci/shopping_queries_dataset_products.parquet
	$(PY) -m bench.esci_prep --src data/esci --out data/esci_smoke --embedder hash --limit 300000 --queries-per-locale 400
	$(PY) -m bench.runner --config configs/esci_smoke.yaml
	$(PY) -m bench.esci_eval --run results/esci_smoke --emb data/esci_smoke

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
	$(PY) -m jupyter nbconvert --to pdf notebooks/results.ipynb --output-dir results

all: sift1m esci notebook

clean-results:
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p) if os.path.isdir(p) else os.remove(p) for p in glob.glob('results/*')]"
