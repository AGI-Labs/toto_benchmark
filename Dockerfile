FROM nvidia/cuda:11.4.2-base-ubuntu20.04 

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

# miniconda
RUN apt install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
SHELL ["/bin/bash", "-c"]
RUN conda init bash


# # install essentials
RUN apt install -y build-essential cmake libtinfo-dev lsb-release libmpfr-dev libedit-dev freeglut3 \
    freeglut3-dev libxmu-dev libxmu-headers libhiredis-dev unzip libnuma-dev


# install Xvfb
RUN apt install -y xvfb
RUN apt install -y time

# install mamba/boa into conda `base` environment
RUN conda install mamba -y -c conda-forge

# unzip the file
RUN mkdir TOTO_starter
COPY ./ /TOTO_starter/
RUN ls -la /TOTO_starter/*
# WORKDIR TOTO_starter
WORKDIR /TOTO_starter

RUN conda env create -f ./environment.yml

## ADD CONDA ENV PATH TO LINUX PATH 
ENV PATH /root/miniconda3/envs/robocloud-env/bin:$PATH
ENV CONDA_DEFAULT_ENV robocloud-env
# make sure to put your env name in place of myconda

## MAKE ALL BELOW RUN COMMANDS USE THE NEW CONDA ENVIRONMENT
SHELL ["conda", "run", "-n", "robocloud-env", "/bin/bash", "-c"]

RUN mamba install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

WORKDIR /
RUN git clone https://github.com/AGI-Labs/robot_baselines.git
RUN pip install -e ./robot_baselines
RUN pip install git+https://github.com/openai/CLIP.git

RUN echo "conda activate robocloud-env" >> ~/.bashrc