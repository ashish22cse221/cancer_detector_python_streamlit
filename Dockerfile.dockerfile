# ── Thermo Spectroscope — Docker Image ────────────────────────────────────────
FROM python:3.11-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY app.py .

# Model files — must be present at build time
COPY svm_cancer_model.pkl .
COPY scaler.pkl .
COPY pca.pkl .

# Streamlit environment
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]
```

---

**`requirements.txt`** (place alongside the Dockerfile)
```
streamlit>=1.35.0
numpy>=1.24.0
joblib>=1.3.0
opencv-python-headless>=4.8.0
matplotlib>=3.7.0
Pillow>=10.0.0
scikit-learn>=1.3.0