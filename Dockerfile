FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY frontend/ ./frontend/

# 仅作文档说明用，Render 实际会通过 PORT 环境变量分配端口
EXPOSE 7860

# 用 shell 形式启动，这样 ${PORT} 才会被展开；本地没设 PORT 时退回 7860
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
