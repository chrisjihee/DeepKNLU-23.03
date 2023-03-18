# 1. reset
export PROJECT_NAME="DeepKorean-23.03"
export PYTHON_VER="3.10"
export CUDA_VER="11.7"
conda update -n base -c defaults conda -y
conda create -n $PROJECT_NAME python=$PYTHON_VER -y
conda activate $PROJECT_NAME
conda install cuda-nvcc=$CUDA_VER cudatoolkit=$CUDA_VER -c nvidia -y

# 2. uneditable library
pip install --upgrade torch deepspeed evaluate datasets tokenizers
pip install --upgrade matplotlib notebook ipython ipynbname jupyterlab tornado==6.1
pip list --format=freeze >requirements.txt
pip install -r requirements.txt

# 3. editable library
rm -rf transformers lightning chrisdict chrisbase chrislab ratsnlp
git clone git@github.com:chrisjihee/transformers.git
git clone git@github.com:chrisjihee/lightning.git
git clone git@github.com:chrisjihee/chrisdict.git
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrislab.git
git clone git@github.com:chrisjihee/ratsnlp.git
pip install --editable transformers
pip install --editable lightning[extra]
pip install --editable chrisdict
pip install --editable chrisbase
pip install --editable chrislab
pip install --editable ratsnlp

# 4. resource
mkdir -p data model
git clone guest@129.254.164.137:git/data-korquad data/korquad
git clone guest@129.254.164.137:git/pretrained-com model/pretrained
