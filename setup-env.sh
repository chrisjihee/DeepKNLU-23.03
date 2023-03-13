# 0. ready
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.profile
source ~/.bashrc

# 1. reset
conda update -n base -c defaults conda
conda create -n DeepKorean-23.03 python=3.10 -y
conda activate DeepKorean-23.03

# 2. install
conda install cuda-nvcc=11.7 cudatoolkit=11.7 -c nvidia -y
#pip install -r requirements.txt
pip install --upgrade torch torchvision torchaudio
pip install --upgrade deepspeed
pip install --upgrade lightning
pip install --upgrade datasets evaluate transformers
pip install --upgrade chrisbase chrisdict chrislab
pip install --upgrade matplotlib ipython ipynbname nb_extension_tagstyler jupyter
pip list --format=freeze >requirements.txt
pip install --upgrade ratsnlp

# 3. config
rm -rf .jupyter ~/.cache/huggingface
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --generate-config
ln -s ~/.jupyter config/jupyter
cp config/jupyter_notebook_config.py ~/.jupyter/

# 3. clone
mkdir -p data
git clone guest@129.254.164.137:git/PretrainedLM pretrained
git clone guest@129.254.164.137:git/data-korquad data/korquad
