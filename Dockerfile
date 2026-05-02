FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm + shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# churn_artifacts_v1.pkl must be present at build time or mounted
# Railway: add it via Railway Volume or build it into the image

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]