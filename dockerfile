FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG python=3.8
ENV PYTHON_VERSION=${python}

RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    libhdf5-dev \
    libsndfile1 \
    ffmpeg \
    wget \
    git \
    curl

RUN pip3 install --upgrade pip setuptools wheel


RUN pip3 install --no-cache-dir --default-timeout=1000 fastapi
RUN pip3 install --no-cache-dir --default-timeout=1000 uvicorn
RUN pip3 install --no-cache-dir --default-timeout=1000 faiss-cpu
RUN pip3 install --no-cache-dir --default-timeout=1000 sentence-transformers
RUN pip3 install --no-cache-dir --default-timeout=1000 pytube
RUN pip3 install --no-cache-dir --default-timeout=1000 yt-dlp
RUN pip3 install --no-cache-dir --default-timeout=1000 opencv-python-headless
RUN pip3 install --no-cache-dir --default-timeout=1000 PIL
RUN pip3 install --no-cache-dir --default-timeout=1000 streamlit
RUN pip3 install --no-cache-dir --default-timeout=1000 openai-whisper
RUN pip3 install --no-cache-dir --default-timeout=1000 python-multipart

# Expose the FastAPI application port
EXPOSE 8080

# Define the entrypoint for the application
ENTRYPOINT ["uvicorn", "app:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--host", "0.0.0.0", "--port", "8080", "--preload", "--timeout", "600"]