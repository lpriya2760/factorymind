FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit ./.streamlit
COPY factorymind ./factorymind
COPY app.py .

# Cloud Run sets PORT.
CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
