# 1. Use Python 3.11 slim as the base
FROM python:3.11-slim

# 2. Set environment variables
# Prevents Python from writing .pyc files and ensures logs are sent straight to terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# 3. Install system dependencies
# Added 'gcc' and 'python3-dev' which are often required for building
# medical imaging C-extensions in Python 3.11
RUN apt-get update && apt-get install -y --no-install-packages-hint \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
# Upgrade pip first to ensure compatibility with 3.11 wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY . .

# 6. Create necessary directory structure
RUN mkdir -p data/raw data/processed data/outputs logs models static/css static/images

# 7. Expose Streamlit port
EXPOSE 8501

# 8. Optimized Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]