FROM ubuntu:22.04 AS base

#ARG UNAME
#ARG UID
#ARG GID

# These get overwritten by values in compose.yaml
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install system dependencies & PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    libx11-6 \
    libxi6 \
    libxxf86vm1 \
    libxfixes3 \
    libxrender1 \
    libgl1-mesa-glx \
    libxkbcommon0 \
    libsm6 \
    libglib2.0-0 \
    wget \
    xz-utils \
    nano \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update

# Install Blender
#RUN wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz \
#    && tar -xvf blender-4.2.0-linux-x64.tar.xz -C /opt/ \
#    && ln -s /opt/blender-4.2.0-linux-x64/blender /usr/local/bin/blender \
#    && rm blender-4.2.0-linux-x64.tar.xz

# Install Python version and build tools
# Note: python3.12-distutils is often needed for pip on deadsnakes
RUN apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    coinor-libipopt-dev \
    gcc g++ gfortran git patch pkg-config \
    liblapack-dev libmetis-dev cmake vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python version as the default 'python' and 'python3'
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install and upgrade pip for python version
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

RUN python -m pip install -U pip && python -m pip install -U opencv-python numpy Pillow matplotlib scipy natsort h5py seaborn ipython matplotlib scipy marimo jax cvxpy equinox optax datetime pandas scikit-image pytest

RUN python -m pip install --upgrade "jax[cuda${CUDA_VERSION}]"

# Install moreau
RUN python -m pip install moreau[cuda${CUDA_VERSION}] --extra-index-url https://kkMeu-DhcjKJQMlfKzSqkxXqjEQndqjgE:@pypi.fury.io/optimalintellect/

# Silence the missing group name warning for GID 992
RUN getent group 992 || groupadd -g 992 host_group_992

# Create a user and group matching the host to avoid volume permission issues
#RUN getent group ${GID} || groupadd -g ${GID} ${UNAME} \
#    && id -u ${UID} || useradd -u ${UID} -g ${GID} -m -s /bin/bash ${UNAME}

# Give the new user ownership of the workspace directory
#RUN chown -R ${UID}:${GID} /workspace

#USER ${UNAME}

CMD ["/bin/bash"]
