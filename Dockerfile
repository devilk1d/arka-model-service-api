FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm + shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# churn_artifacts_v2.pkl must be present at build time or mounted via Railway Volume
# v2.1: single artifact file — tidak ada lagi nlp_artifacts_v1.pkl

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]