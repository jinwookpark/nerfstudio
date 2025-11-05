# syntax=docker/dockerfile:1

ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.4.1
# CUDA architectures used by COLMAP / TCNN / gsplat
ARG CUDA_ARCHITECTURES="86;80;75;70"
ARG NERFSTUDIO_VERSION="v1.1.5"

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# --------- ARGS / ENVS ---------
ARG CUDA_ARCHITECTURES
ARG NVIDIA_CUDA_VERSION
ARG UBUNTU_VERSION
ARG NERFSTUDIO_VERSION

# non-root user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"
# 필요시 pip 캐시 비활성화
ENV PIP_NO_CACHE_DIR=1

# --------- BASE PACKAGES ---------
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git wget curl ca-certificates \
        ninja-build build-essential pkg-config \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        python3.10-dev python3-pip \
        python-is-python3 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# --------- CMAKE (신버전) ---------
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.31.3/cmake-3.31.3-linux-x86_64.sh -O /tmp/cmake-install.sh \
 && chmod +x /tmp/cmake-install.sh \
 && mkdir -p /opt/cmake-3.31.3 \
 && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.3 \
 && rm -f /tmp/cmake-install.sh \
 && ln -s /opt/cmake-3.31.3/bin/* /usr/local/bin

# --------- GLOMAP ---------
RUN git clone https://github.com/colmap/glomap.git /tmp/glomap && \
    cd /tmp/glomap && git checkout "1.0.0" && \
    mkdir -p build && cd build && \
    cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja -j$(nproc) && ninja install && \
    cd / && rm -rf /tmp/glomap

# --------- COLMAP ---------
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && git checkout "3.12.6" && \
    mkdir -p build && cd build && \
    cmake .. -GNinja "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja -j$(nproc) && ninja install && \
    cd / && rm -rf /tmp/colmap

# --------- USER ---------
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -d /home/${USERNAME} --shell /usr/bin/bash \
 && echo "${USERNAME}:password" | chpasswd \
 && usermod -aG sudo ${USERNAME} \
 && echo "%sudo ALL=NOPASSWD:/usr/bin/apt-get update, /usr/bin/apt-get upgrade, /usr/bin/apt-get install, /usr/bin/apt-get remove" >> /etc/sudoers \
 && mkdir -p /workspace && chown ${USER_UID}:${USER_GID} /workspace

USER ${USER_UID}
WORKDIR /home/${USERNAME}


# --------- PYTHON + TORCH ---------
# 빌드 안정화를 위한 wheel/ninja/packaging 선설치
RUN pip install --upgrade pip "setuptools<70.0.0" wheel ninja packaging && \
    pip install \
      torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
      --index-url https://download.pytorch.org/whl/cu124

# --------- HLOC (수정 브랜치) ---------
RUN git clone --branch pjw_hloc --recursive https://github.com/jinwookpark/Hierarchical-Localization.git && \
    cd Hierarchical-Localization && python3.10 -m pip install -e . && cd ..

# --------- TCNN (tiny-cuda-nn) ---------
# Torch/CUDA 아키텍처 전달 (둘 다 설정)
ENV TCNN_COMMIT=b3473c81396fe927293bdfd5a6be32df8769927c
ENV TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
# "86;80;75;70" -> "8.6 8.0 7.5 7.0"
RUN export TORCH_CUDA_ARCH_LIST="$(echo "${CUDA_ARCHITECTURES}" | tr ';' ' ' | sed 's/\([0-9]\)\([0-9]\)/\1.\2/g')" && \
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" && \
    echo "TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}" && \
    python3 -m pip install -U pybind11 && \
    git clone https://github.com/NVlabs/tiny-cuda-nn.git /tmp/tcnn && \
    cd /tmp/tcnn && git checkout ${TCNN_COMMIT} && git submodule update --init --recursive && \
    # 병렬 빌드 안정화(메모리 부족시 1~2로 낮추세요)
    export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 && \
    # build-isolation 끄고 현재 torch 사용, verbose로 로그 확인
    python3 -m pip install --no-build-isolation -vv ./bindings/torch && \
    cd / && rm -rf /tmp/tcnn

# --------- pycolmap / pyceres / omegaconf ---------
RUN pip install pycolmap==3.12.3 pyceres==2.1 omegaconf==2.3.0

# Copy nerfstudio folder and give ownership to user.
COPY --chown=${USER_UID}:${USER_GID} . /home/${USERNAME}/nerfstudio

# --------- gsplat (fork/branch pjw) 선 설치 ---------
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 > 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    export MAX_JOBS=4 && \
    #GSPLAT_VERSION="$(sed -n 's/.*gsplat==\s*\([^," '"'"']*\).*/\1/p' /home/${USERNAME}/nerfstudio/pyproject.toml)" && \
    #pip install --no-cache-dir git+https://github.com/nerfstudio-project/gsplat.git@v${GSPLAT_VERSION} && \
    pip install --no-cache-dir -e /home/${USERNAME}/nerfstudio 'numpy<2.0.0' && \
    cd ~

# nerfstudio CLI 자동완성
RUN /bin/bash -lc 'ns-install-cli --mode install'

# --------- ENTRY ---------
CMD ["bash","-l"]

