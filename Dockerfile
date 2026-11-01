FROM python:3.12-slim

# docker CLI (memory sampling via `docker stats`), curl for downloads, make
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl make ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["make", "help"]
