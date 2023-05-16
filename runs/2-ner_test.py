from chrisbase.io import ProjectEnv

from nlpbook.arguments import NLUTesterArguments, DataFiles
from nlpbook.ner import cli

config = NLUTesterArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    downstream_data_file=DataFiles(test="valid.txt"),
    downstream_data_name="kmou-ner",
    downstream_data_home="data",
    downstream_data_caching=False,
    downstream_model_file=None,
    downstream_model_home="model/finetuned/kmou-ner-" + "0516",
    pretrained_model_path="model/pretrained-com/KcBERT-Base",
    max_seq_length=50,
    batch_size=100,
    accelerator="gpu",
    precision=16,
).save_working_config()

cli.test(config)
