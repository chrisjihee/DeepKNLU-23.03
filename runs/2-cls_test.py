from nlpbook.arguments import *
from nlpbook.cls import cli

args = TesterArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    model=ModelArgs(
        data_file=DataFiles(test="ratings_test.txt"),
        data_name="nsmc",
        data_home="data",
        data_caching=False,
        finetuned_name=None,
        finetuned_home="model/finetuned",
        pretrained_name="model/pretrained-com/KcBERT-Base",
        max_seq_length=50,
    ),
    hardware=HardwareArgs(
        batch_size=100,
        accelerator="gpu",
        precision=16,
    ),
)
with UsingArguments(args) as args_file:
    cli.test(args_file)
