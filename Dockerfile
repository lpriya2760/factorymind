FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# XGBoost needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit ./.streamlit
COPY factorymind ./factorymind
COPY app.py .
COPY demo_bundles/*.zip ./demo_bundles/

# Cloud Run sets PORT.
CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
