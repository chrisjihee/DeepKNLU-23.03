from nlpbook.arguments import *
from nlpbook.cls import cli

args = TesterArguments(
    env=ProjectEnv(project="DeepKNLU", running_gpus="0"),
    data=DataOption(
        home="data",
        name="nsmc",
        files=DataFiles(test="ratings_test.txt"),
        caching=False,
    ),
    model=ModelOption(
        pretrained="pretrained-com/KcBERT-Base",
        finetuning_home="finetuning",
        finetuning_name=None,
        max_seq_length=50,
    ),
    hardware=HardwareOption(
        accelerator="gpu",
        batch_size=100,
        precision=16,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.test(args_file)
