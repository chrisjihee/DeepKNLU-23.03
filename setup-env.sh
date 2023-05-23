#!/bin/bash
# Tested on Ubuntu 20.04
# 1. uneditable library
export PROJECT_NAME="DeepKorNLU-23.03"
export PYTHON_VER="3.10"
export CUDA_VER="11.7"
conda update -n base -c defaults conda -y
conda create -n $PROJECT_NAME python=$PYTHON_VER -y
conda activate $PROJECT_NAME
if [ $(uname -s) = "Linux" ]; then
  conda install cuda-nvcc=$CUDA_VER cudatoolkit=$CUDA_VER -c nvidia -y
  pip install torch --index-url https://download.pytorch.org/whl/cu117
fi
pip install torch

# 2. editable library
rm -rf transformers lightning chrisbase chrislab
git clone git@github.com:huggingface/transformers.git -b v4.29.2
git clone git@github.com:Lightning-AI/lightning.git -b 2.0.2
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrislab.git
pip install --editable transformers
pip install --editable lightning
pip install --editable chrisbase
pip install --editable chrislab

# 3. pretrained model
mkdir -p model/pretrained
cd model/pretrained
git lfs install
git clone https://huggingface.co/beomi/kcbert-base KcBERT-Base
#git clone https://huggingface.co/beomi/KcELECTRA-base-v2022 KcELECTRA-Base
#git clone https://huggingface.co/skt/kobert-base-v1 KoBERT-Base
#git clone https://huggingface.co/monologg/koelectra-base-v3-discriminator KoELECTRA-Base
#git clone https://github.com/KPFBERT/kpfbert KPF-BERT-Base
#git clone https://huggingface.co/klue/bert-base KLUE-BERT-Base
#git clone https://huggingface.co/klue/roberta-base KLUE-RoBERTa-Base
#git clone https://huggingface.co/bert-base-multilingual-uncased Google-BERT-Base
#git clone https://huggingface.co/monologg/kobigbird-bert-base KoBigBird-Base
git lfs uninstall
cd ../..

# 3. pretrained model (private)
#ln -s /dat/proj/pretrained-com model
#ln -s /dat/proj/pretrained-pro model

# 4. eval
pip install overrides==3.1.0
pip install importlib-metadata==4.13.0
