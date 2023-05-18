from nlpbook.arguments import *
from nlpbook.ner import cli

args = ServerArguments(
    env=ProjectEnv(project="DeepKorNLU"),
    model=ModelArgs(
        data_name="kmou-ner",
        finetuned_name=None,
        finetuned_home="model/finetuned",
        pretrained_name="model/pretrained-com/KcBERT-Base",
        max_seq_length=50,
    ),
)
with UsingArguments(args) as args_file:
    cli.serve(args_file)
