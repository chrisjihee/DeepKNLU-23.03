from chrisbase.io import ProjectEnv
from chrislab.ratsnlp import cli
from ratsnlp.nlpbook.arguments import NLUServerArguments

config = NLUServerArguments(
    env=ProjectEnv(project="DeepKorNLU"),
    pretrained_model_path="model/pretrained-com/KcBERT-Base",
    downstream_model_home=f"model/finetuned/kmou-ner-0512",
    downstream_model_file=None,
    max_seq_length=50,
).save_working_config()

cli.serve_ner(config)
