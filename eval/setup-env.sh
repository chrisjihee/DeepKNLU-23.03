#!/bin/bash
# Tested on Ubuntu 20.04
export PROJECT_NAME="DeepKorNLU-klue"
export PYTHON_VER="3.7"
conda update -n base -c defaults conda -y
conda create -n $PROJECT_NAME python=$PYTHON_VER -y
conda activate $PROJECT_NAME

pip install dataclasses
pip install torch==1.7.0 --index-url https://download.pytorch.org/whl/cu110
pip install -r requirements.txt

