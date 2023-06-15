#!/bin/bash
# WARNING: should unset LD_LIBRARY_PATH before run python script

# 1. uneditable library
export PROJECT_NAME="DeepKNLU-23.03"
export PYTHON_VER="3.10"
export CUDA_NVCC_VER="11.8"
export TORCH_URL="https://download.pytorch.org/whl/cu118"
conda update -n base -c conda-forge conda -y
conda create -n $PROJECT_NAME python=$PYTHON_VER -y
conda activate $PROJECT_NAME
if [ "$(uname)" == "Darwin" ]; then
  conda install pytorch::pytorch -c pytorch
elif command -v nvidia-smi; then
  conda install -y -c nvidia cuda-nvcc=$CUDA_NVCC_VER
  conda install -y -c nvidia cudatoolkit=$CUDA_NVCC_VER
  conda install -y -c nvidia cudnn
  pip install torch --index-url $TORCH_URL
else
  pip install torch
fi
conda list

# 2. editable library
rm -rf transformers lightning chrisbase chrislab
git clone git@github.com:huggingface/transformers.git -b v4.30.2
git clone git@github.com:Lightning-AI/lightning.git -b 2.0.3
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrislab.git
pip install --editable transformers
pip install --editable lightning
pip install --editable chrisbase
pip install --editable chrislab

# 3. pretrained model (option 1)
#git clone guest@129.254.164.137:git/pretrained-com
#git clone chris@129.254.164.137:git/pretrained-pro
ln -s ../pretrained-com
ln -s ../pretrained-pro

# 3. pretrained model (option 2)
mkdir -p pretrained
cd pretrained || return
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
