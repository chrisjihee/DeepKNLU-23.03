from nlpbook.arguments import *
from nlpbook.ner import cli

args = ServerArguments(
    env=ProjectEnv(project="DeepKorNLU"),
    data=DataOption(name="kmou-ner"),
    model=ModelOption(
        pretrained="model/pretrained-com/KcBERT-Base",
        finetuning_home="model/finetuning",
        max_seq_length=50,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.serve(args_file)
