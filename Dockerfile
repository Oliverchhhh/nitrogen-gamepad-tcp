FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=http://127.0.0.1:58931
ENV https_proxy=http://127.0.0.1:58931

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    pkg-config \
    clang \
    libclang-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    libavfilter-dev \
    socat \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY . .

RUN uv python install 3.13.2 && \
    ulimit -n 65535 && \
    uv sync

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["bash"]
