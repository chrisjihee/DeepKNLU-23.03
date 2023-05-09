from chrisbase.io import ProjectEnv
from chrislab.ratsnlp import cli
from ratsnlp.nlpbook.arguments import NLUServerArguments

config = NLUServerArguments(
    env=ProjectEnv(project="DeepKorNLU"),
    pretrained_model_path="model/pretrained/KcBERT-Base",
    downstream_model_home="model/finetuned/nsmc-0509",
    downstream_model_file=None,
    max_seq_length=64,
).save_working_config()

cli.serve_cls(config)
