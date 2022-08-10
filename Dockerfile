# Base container that includes all dependencies but not the actual repo

ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=11.3

FROM nvidia/cudagl${ARCH:+-$ARCH}:${CUDA}.0-base-ubuntu${UBUNTU_VERSION} as base

ARG UBUNTU_VERSION
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.5.32-1

SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND="noninteractive"
# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
    
# NOTE: we don't use TF so might not need some of these
# ========== Tensorflow dependencies ==========
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        #cuda-cublas-${CUDA/./-} \
        #cuda-cufft-${CUDA/./-} \
        #cuda-curand-${CUDA/./-} \
        #cuda-cusolver-${CUDA/./-} \
        #cuda-cusparse-${CUDA/./-} \
        curl \
        # libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        zip \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y
RUN apt-get update --fix-missing
RUN apt-get install -y wget bzip2 ca-certificates git vim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        premake4 \
        git \
        curl \
        vim \
        ffmpeg \
	    libgl1-mesa-dev \
	    libgl1-mesa-glx \
	    libglew-dev \
	    libosmesa6-dev \
	    libxrender-dev \
	    libsm6 libxext6 \
        unzip \
        patchelf \
        ffmpeg \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        python3-dev python3-pip graphviz \
        freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglew1.6-dev mesa-utils
        
# Not sure why this is needed
ENV LANG C.UTF-8
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y
RUN apt update
RUN apt install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y

RUN conda create --name env-rlkit python=3.7 pip
RUN echo "source activate env-rlkit" >> ~/.bashrc
ENV PATH /opt/conda/envs/env-rlkit/bin:$PATH

#RUN mkdir /root/playground

# make sure your domain is accepted
RUN mkdir /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

WORKDIR /root/playground/
RUN git clone https://github.com/XinyuR1/rlkit.git
WORKDIR /root/playground/rlkit
RUN pip install -r requirements.txt
RUN pip install -e ./

WORKDIR /root/playground/
RUN git clone https://github.com/XinyuR1/doodad.git
WORKDIR /root/playground/doodad
RUN pip install -r requirements.txt
RUN pip install -e ./

WORKDIR /root/playground/rlkit
RUN conda list

#CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"