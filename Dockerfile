FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential htop git python3-onnx rdfind

WORKDIR /app

# Add the application files
COPY . /app/

# Initialize and update git submodules
RUN cd /app && \
    git init && \
    git submodule init && \
    git submodule update --init --recursive && \
    git submodule update --recursive && \
    rm -rf .git */.git **/.git  # Remove all .git directories

# Setup conda and create Python 3.10 environment
RUN conda config --set always_yes true && conda init
RUN conda create -n py310 python=3.10
RUN echo "conda activate py310" >> ~/.bashrc

# Install PyTorch and torchvision
RUN conda run -n py310 conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Kaolin dependencies first
RUN conda run -n py310 pip install -r https://raw.githubusercontent.com/NVIDIAGameWorks/kaolin/v0.17.0/tools/build_requirements.txt \
    -r https://raw.githubusercontent.com/NVIDIAGameWorks/kaolin/v0.17.0/tools/viz_requirements.txt \
    -r https://raw.githubusercontent.com/NVIDIAGameWorks/kaolin/v0.17.0/tools/requirements.txt

# Now install Kaolin with the correct version for CUDA 11.8
RUN conda run -n py310 pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html

# Install diso and other dependencies
RUN conda run -n py310 pip install diso

# Verify Kaolin installation
RUN conda run -n py310 python -c "import kaolin; print(kaolin.__version__)"

# Create a g++ wrapper for JIT
RUN printf '#!/usr/bin/env bash\nexec /usr/bin/g++ -I/usr/local/cuda/include -I/usr/local/cuda/include/crt "$@"\n' > /usr/local/bin/gxx-wrapper && \
    chmod +x /usr/local/bin/gxx-wrapper
ENV CXX=/usr/local/bin/gxx-wrapper

# Run setup.sh with Python 3.10
RUN conda run -n py310 ./setup.sh --basic --xformers --flash-attn --diffoctreerast --vox2seq --spconv --mipgaussian --kaolin --nvdiffrast --demo

# Install additional Python packages
RUN conda run -n py310 pip install plyfile utils3d flash_attn spconv-cu120 xformers
RUN conda run -n py310 pip install git+https://github.com/NVlabs/nvdiffrast.git

# Install optree
RUN conda run -n py310 pip install 'optree>=0.13.0'

# Cleanup
RUN apt-get remove -y ffmpeg build-essential htop git python3-onnx && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda clean --all -f -y
RUN rdfind -makesymlinks true /opt/conda

# Final stage
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel AS final

WORKDIR /app
COPY --from=builder /usr/local/bin/gxx-wrapper /usr/local/bin/gxx-wrapper
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root /root
COPY --from=builder /app /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y build-essential \
                       git \
                       strace \
                       vim && \
    rm -rf /var/lib/apt/lists/*

# Add FastAPI dependencies
RUN conda run -n py310 pip install fastapi uvicorn python-multipart

# Add the startup script
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Set environment variables
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=py310
ENV CONDA_PREFIX=/opt/conda/envs/py310

# Activate Python 3.10 environment by default
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate py310" >> ~/.bashrc

CMD ["/app/startup.sh"]
