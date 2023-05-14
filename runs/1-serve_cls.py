from chrisbase.io import ProjectEnv
from nlpbook.arguments import NLUServerArguments
from nlpbook.cls import cli

config = NLUServerArguments(
    env=ProjectEnv(project="DeepKorNLU"),
    pretrained_model_path="model/pretrained-com/KcBERT-Base",
    downstream_model_home="model/finetuned/nsmc-0512",
    downstream_model_file=None,
    max_seq_length=50,
).save_working_config()

cli.serve_cls(config)
