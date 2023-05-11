#!/bin/bash
# Tested on Ubuntu 20.04
export PROJECT_NAME="DeepKorNLU-klue"
export PYTHON_VER="3.7"
conda update -n base -c defaults conda -y
conda create -n $PROJECT_NAME python=$PYTHON_VER -y
conda activate $PROJECT_NAME
pip install -r requirements.txt

