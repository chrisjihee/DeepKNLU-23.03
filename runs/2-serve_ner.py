from chrisbase.io import ProjectEnv
from nlpbook.arguments import NLUServerArguments
from nlpbook.ner import cli

config = NLUServerArguments(
    env=ProjectEnv(project="DeepKorNLU"),
    pretrained_model_path="model/pretrained-com/KcBERT-Base",
    downstream_model_home=f"model/finetuned/kmou-ner-0514",
    downstream_model_file=None,
    downstream_task_name="ner",
    max_seq_length=50,
).save_working_config()

cli.serve_ner(config)
