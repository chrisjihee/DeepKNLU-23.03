from nlpbook.arguments import *
from nlpbook.ner import cli

args = ServerArguments(
    env=ProjectEnv(project="DeepKNLU"),
    data=DataOption(name="kmou-ner"),
    model=ModelOption(
        pretrained="pretrained-com/KcBERT-Base",
        finetuning_home="finetuning",
        max_seq_length=50,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.serve(args_file)
