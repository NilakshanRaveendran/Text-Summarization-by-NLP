FROM python:3.9-slim

# Avoid CUDA/Triton downloads
ENV FORCE_CUDA=0

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install Python deps
# HuggingFace (CPU only) 
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers accelerate

CMD ["python3", "app.py"]