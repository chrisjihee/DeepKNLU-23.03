from chrisbase.io import ProjectEnv
from chrisbase.time import now
from chrislab.ratsnlp import cli
from ratsnlp.nlpbook.arguments import NLUTrainerArguments

config = NLUTrainerArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    pretrained_model_path="model/pretrained/KcBERT-Base",
    downstream_model_home=f"model/finetuned/ner-{now('%m%d')}",
    downstream_model_file="{epoch}-{val_loss:.3f}-{val_acc:.3f}",
    downstream_task_name="ner",
    downstream_data_home="data",
    downstream_data_name="ner",
    learning_rate=5e-5,
    max_seq_length=50,
    batch_size=100,
    precision="16-mixed",
    epochs=1,
    seed=7,
).save_working_config()

cli.train_ner(config)
