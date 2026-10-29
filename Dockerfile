# ==============================================================================
# Dockerfile: Turnkey Artifact Evaluation Environment for SMR-Engine
# Tailored for NVIDIA Blackwell / Ada GPUs (GeForce RTX 5070 Ti / RTX 4090)
# Venue: MLSys Artifact Evaluation
# ==============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, Linux perf tools, build essentials, and ARM toolchain for MCU audit
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    linux-tools-generic \
    libnuma-dev \
    gcc-arm-none-eabi \
    libnewlib-arm-none-eabi \
    qemu-system-arm \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8 support and systems monitoring toolchain
RUN python -m pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 || \
    python -m pip install --no-cache-dir torch torchvision torchaudio

# Install systems benchmarking and profiling libraries
RUN python -m pip install --no-cache-dir \
    pynvml \
    nvidia-ml-py \
    psutil \
    numpy \
    scipy \
    matplotlib \
    transformers \
    accelerate \
    timm \
    tqdm

# Create workspace directory
WORKDIR /workspace/SMR-Engine

# Copy repository contents
COPY . /workspace/SMR-Engine/

# Ensure bash scripts are executable
RUN python setup.py build_ext --inplace || true
RUN chmod +x run_all_benchmarks.sh

# Default entrypoint: Run the master reproduction harness
CMD ["bash", "run_all_benchmarks.sh"]
