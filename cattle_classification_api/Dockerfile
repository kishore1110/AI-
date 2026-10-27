# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Ensure Python output is sent straight to the container logs (no buffering).
# Without this, logs may not appear before an OOM crash — making debugging near-impossible.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV (used by ultralytics/YOLO)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight deps first, then heavy ML libs separately to limit peak RAM
RUN pip install --no-cache-dir fastapi==0.110.0 uvicorn==0.27.1 python-multipart==0.0.9 Pillow==10.2.0
# Prometheus monitoring — tiny package, install early so it is cached separately
RUN pip install --no-cache-dir prometheus-fastapi-instrumentator==6.1.0
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.1+cpu torchvision==0.17.1+cpu
RUN pip install --no-cache-dir ultralytics==8.1.24
# Reinstall numpy LAST so no other package can silently downgrade/replace it.
# numpy 1.26.4 is the final stable 1.x release and is fully compatible with torch 2.2.x
RUN pip install --no-cache-dir "numpy==1.26.4"

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Change working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app
COPY --chown=user . $HOME/app

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# Run with 1 worker to minimize memory footprint.
# --timeout-keep-alive 75 matches HF's proxy timeout to avoid premature 502s.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "75"]
