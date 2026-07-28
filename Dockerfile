# syntax=docker/dockerfile:1
FROM python:3.11-slim

# OpenCV needs these system libraries even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY tests/ tests/

# Mount your dataset at /app/data/chest_xray and artifacts at /app/artifacts, e.g.:
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts \
#     pneumonia-xray-cnn --model improved --epochs 10
ENTRYPOINT ["python", "-m", "pneumonia_cnn.train"]
CMD ["--help"]
