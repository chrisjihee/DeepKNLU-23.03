from chrisbase.io import ProjectEnv
from nlpbook.arguments import NLUTesterArguments
from nlpbook.ner import cli

config = NLUTesterArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    pretrained_model_path="model/pretrained-com/KcBERT-Base",
    downstream_model_home=f"model/finetuned/kmou-ner-0514",
    downstream_model_file=None,
    downstream_task_name="ner",
    downstream_data_home="data",
    downstream_data_name="kmou-ner",
    downstream_data_file="val.txt",
    max_seq_length=50,
    batch_size=100,
    accelerator="gpu",
    precision=16,
    strategy="single_device",
    devices=1,
).save_working_config()

cli.test_ner(config)
