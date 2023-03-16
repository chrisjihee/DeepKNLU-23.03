# 0. ready
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.profile
source ~/.bashrc
conda update -n base -c defaults conda

# 1. reset
conda create -n DeepKorean-23.03 python=3.10 -y
conda activate DeepKorean-23.03

# 2. install
conda install cuda-nvcc=11.7 cudatoolkit=11.7 -c nvidia -y
#pip install -r requirements.txt
pip install --upgrade torch
pip install --upgrade deepspeed
pip install --upgrade lightning
pip install --upgrade lightning[extra]
pip install --upgrade evaluate datasets tokenizers #transformers
pip install --upgrade ipython ipynbname nb_extension_tagstyler jupyter
pip install --upgrade matplotlib
pip list --format=freeze >requirements.txt
git clone git@github.com:huggingface/transformers.git -b v4.27.1 transformers
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrisdict.git
git clone git@github.com:chrisjihee/chrislab.git
git clone git@github.com:chrisjihee/ratsnlp.git
pip install --editable transformers
pip install --editable chrisdict
pip install --editable chrisbase
pip install --editable chrislab
pip install --editable ratsnlp

# 3. config
rm -rf .jupyter ~/.cache/huggingface
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --generate-config -y
ln -s ~/.jupyter config/jupyter
cp config/jupyter_notebook_config.py ~/.jupyter/

# 3. clone
mkdir -p data
git clone guest@129.254.164.137:git/PretrainedLM pretrained
git clone guest@129.254.164.137:git/data-korquad data/korquad
