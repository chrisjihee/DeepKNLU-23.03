from chrisbase.io import ProjectEnv
from nlpbook.arguments import NLUServerArguments
from nlpbook.cls import cli

config = NLUServerArguments(
    env=ProjectEnv(project="DeepKorNLU"),
    pretrained_model_path="model/pretrained-com/KcBERT-Base",
    downstream_model_home="model/finetuned/nsmc-" + "0516",
    downstream_model_file=None,
    downstream_task_name="cls",
    max_seq_length=50,
).save_working_config()

cli.serve(config)
