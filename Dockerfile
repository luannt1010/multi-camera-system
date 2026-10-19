FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1 TZ=Asia/Ho_Chi_Minh

WORKDIR /app

# Cài FFmpeg và thư viện cho OpenCV.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0t64 libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/*

# Cài dependency Python.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy code và cấu hình.
COPY src/ ./src/
COPY configs/ ./configs/

# Chuẩn bị user và quyền ghi dữ liệu.
RUN groupadd -g 10001 app \
    && useradd -m -u 10001 -g app app \
    && mkdir -p data logs \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "8000"]