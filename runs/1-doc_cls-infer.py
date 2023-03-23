from chrisbase.io import ProjectEnv
from chrislab.ratsnlp import cli
from ratsnlp.nlpbook.classification.arguments import ClassificationDeployArguments

config = ClassificationDeployArguments(
    env=ProjectEnv(project="DeepKorean"),
    pretrained_model_path="model/pretrained/KcBERT-Base",
    downstream_model_home="model/finetuned/nsmc",
    downstream_model_file=None,
    max_seq_length=128,
).save_working_config()

cli.serve(config)
