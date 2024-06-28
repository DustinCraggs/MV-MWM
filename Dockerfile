FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

RUN apt-get update -y
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get -y install python3.8 python3-pip

RUN apt-get update -y && apt-get install -y \
        git \
        vim \
        tmux \
        wget \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxkbcommon-x11-0 \
        mesa-utils

ENV COPPELIASIM_ROOT=/CoppeliaSim
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda

RUN wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
RUN mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
RUN rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
RUN pip install git+https://github.com/stepjam/RLBench.git

WORKDIR /mv_mwm

COPY . .
RUN /mv_mwm/dependency.sh
RUN pip install -e rlbench_shaped_rewards
