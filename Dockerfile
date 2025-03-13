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

# Install basic dependencies from setup.sh
RUN pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers && \
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install Kaolin
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html

# Install xformers
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118

# Install flash-attn
RUN pip install flash-attn

# Install additional dependencies for the app
RUN pip install fastapi uvicorn python-multipart optree>=0.13.0

# Setup CUDA environment for JIT compilation
RUN printf '#!/usr/bin/env bash\nexec /usr/bin/g++ -I/usr/local/cuda/include -I/usr/local/cuda/include/crt "$@"\n' > /usr/local/bin/gxx-wrapper && \
    chmod +x /usr/local/bin/gxx-wrapper
ENV CXX=/usr/local/bin/gxx-wrapper

# Install extension dependencies
RUN mkdir -p /tmp/extensions && \
    # Install diffoctreerast
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast && \
    # Install mip-splatting
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    # Install nvdiffrast
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast && \
    # Install spconv
    pip install spconv-cu118

# Run setup script with required flags
RUN ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# Cleanup and optimize
RUN conda clean -a -y && \
    rm -rf /tmp/extensions

# Set environment variables mentioned in README
ENV ATTN_BACKEND=flash-attn
ENV SPCONV_ALGO=native

# Set the default command
CMD ["python", "headless_app.py"]

# Expose the port that FastAPI runs on
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
