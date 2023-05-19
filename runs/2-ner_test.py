from nlpbook.arguments import *
from nlpbook.ner import cli

args = TesterArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    data=DataOption(
        home="data",
        name="kmou-ner",
        files=DataFiles(test="valid.jsonl"),
        caching=False,
    ),
    model=ModelOption(
        pretrained="model/pretrained-com/KcBERT-Base",
        finetuning_home="model/finetuning",
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
