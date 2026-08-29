FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY frontend/ ./frontend/

# 改为 7860
EXPOSE 7860

# 改为 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]