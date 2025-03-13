# Use PyTorch base image with CUDA 11.8
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Set environment variables early
ENV PYTHONPATH=/app
# If you really need to append to an existing PYTHONPATH, use:
# ENV PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/app"
ENV ATTN_BACKEND=flash-attn
ENV SPCONV_ALGO=native
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV CXX=/usr/local/bin/gxx-wrapper

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

# Ensure setup.sh is executable
RUN chmod +x setup.sh

# Install Kaolin and other critical dependencies
RUN pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html && \
    pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install flash-attn && \
    pip install spconv-cu118 && \
    pip install git+https://github.com/NVlabs/nvdiffrast.git && \
    pip install git+https://github.com/autonomousvision/mip-splatting.git && \
    pip install git+https://github.com/JeffreyXiang/diffoctreerast.git && \
    # Verify installations
    python -c "import torch; import kaolin; import xformers; import nvdiffrast; print(f'PyTorch {torch.__version__}, Kaolin {kaolin.__version__}, xformers {xformers.__version__}, nvdiffrast installed')"

# Run setup.sh with remaining flags
RUN ./setup.sh --basic

# Install FastAPI dependencies
RUN pip install fastapi uvicorn python-multipart optree>=0.13.0

# Cleanup
RUN conda clean -a -y

# Expose the port that FastAPI runs on
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command
CMD ["python", "headless_app.py"]
