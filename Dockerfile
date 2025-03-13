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

# Before making setup.sh executable, we should add shebang
RUN echo '#!/bin/bash' | cat - setup.sh > temp && mv temp setup.sh && \
    chmod +x setup.sh

# Install basic dependencies first
RUN pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers

# Install Kaolin and other critical dependencies
RUN pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html

# Install xformers
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118

# Install spconv
RUN pip install spconv-cu118

# Install nvdiffrast
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3

# Install diff-gaussian-rasterization from mip-splatting
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization && \
    rm -rf /tmp/mip-splatting

# Install diffoctreerast
RUN pip install git+https://github.com/JeffreyXiang/diffoctreerast.git

# Add explicit flash-attn installation before setup.sh
RUN pip install flash-attn && \
    pip install einops  # flash-attn dependency

# Run setup.sh with flash-attn flag
RUN ./setup.sh --basic

# Install FastAPI dependencies
RUN pip install fastapi uvicorn python-multipart optree>=0.13.0

# Replace the verification RUN command with a simpler one
RUN python -c "import torch; import kaolin; import xformers; print(f'PyTorch {torch.__version__}, Kaolin {kaolin.__version__}, xformers {xformers.__version__}')"

# Add a startup script
COPY <<EOF /app/verify_cuda.sh
#!/bin/bash
set -e  # Exit on any error

check_cuda() {
    echo "Checking CUDA setup..."
    if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
        echo "ERROR: CUDA is not available"
        exit 1
    fi
    
    if ! python -c "import torch; assert torch.version.cuda == '11.8', f'Wrong CUDA version: {torch.version.cuda}'"; then
        echo "ERROR: Wrong CUDA version"
        exit 1
    fi
    
    echo "CUDA setup verified successfully"
}

check_dependencies() {
    echo "Checking dependencies..."
    python -c "
import sys
import torch
import kaolin
import xformers
import nvdiffrast
import flash_attn
import spconv
from trellis.representations import Gaussian, MeshExtractResult

versions = {
    'PyTorch': torch.__version__,
    'Kaolin': kaolin.__version__,
    'xformers': xformers.__version__,
    'flash_attn': flash_attn.__version__,
    'spconv': spconv.__version__
}

print('Versions:', versions)

# Verify CUDA setup
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.device_count() > 0, 'No GPU devices found'
device = torch.cuda.current_device()
print(f'Using GPU: {torch.cuda.get_device_name(device)}')"
}

main() {
    check_cuda
    check_dependencies
    echo "All checks passed successfully!"
}

main
EOF
RUN chmod +x /app/verify_cuda.sh

# Cleanup
RUN conda clean -a -y

# Expose the port that FastAPI runs on
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Modify CMD to run verification first
CMD ["/bin/bash", "-c", "./verify_cuda.sh && python headless_app.py"]
