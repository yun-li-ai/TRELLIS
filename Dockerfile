# Use PyTorch base image with CUDA 11.8
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Create temporary directory needed by headless_app.py
RUN mkdir -p /tmp/Trellis-demo

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    build-essential \
    git \
    python3-onnx \
    rdfind \
    ninja-build \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY . /app/

# Initialize git submodules
RUN cd /app && \
    git init && \
    git submodule init && \
    git submodule update --init --recursive && \
    git submodule update --recursive && \
    rm -rf .git */.git **/.git

# Setup CUDA environment for JIT compilation
RUN printf '#!/usr/bin/env bash\nexec /usr/bin/g++ -I/usr/local/cuda/include -I/usr/local/cuda/include/crt "$@"\n' > /usr/local/bin/gxx-wrapper && \
    chmod +x /usr/local/bin/gxx-wrapper
ENV CXX=/usr/local/bin/gxx-wrapper

# Set CUDA architecture flags
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Run setup script with all required flags
RUN ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# Install FastAPI dependencies
RUN pip install fastapi uvicorn python-multipart optree>=0.13.0

# Cleanup
RUN conda clean -a -y

# Set environment variables mentioned in README
ENV ATTN_BACKEND=flash-attn
ENV SPCONV_ALGO=native

# Expose the port that FastAPI runs on
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command
CMD ["python", "headless_app.py"]
