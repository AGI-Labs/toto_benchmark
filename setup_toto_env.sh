#! /bin/sh

# create a new conda environment
mamba env create -f ./toto_benchmark.yml

# activate the environment
conda activate toto-benchmark

# install additional required packages
mamba install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/AGI-Labs/robot_baselines.git
pip install git+https://github.com/openai/CLIP.git