# 1. reset
conda update -n base -c defaults conda -y
conda create -n DeepKorean-23.03 python=3.10 -y
conda activate DeepKorean-23.03

# 2. uneditable library
conda install cuda-nvcc=11.7 cudatoolkit=11.7 -c nvidia -y
pip install --upgrade torch deepspeed evaluate datasets tokenizers
pip install --upgrade notebook ipython ipynbname jupyterlab==3.0.0 tornado==6.1 matplotlib
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
mkdir -p data
git clone guest@129.254.164.137:git/PretrainedLM pretrained
git clone guest@129.254.164.137:git/data-korquad data/korquad
