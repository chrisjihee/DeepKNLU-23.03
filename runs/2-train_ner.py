from chrisbase.io import ProjectEnv
from chrisbase.time import now
from nlpbook.arguments import NLUTrainerArguments
from nlpbook.ner import cli

config = NLUTrainerArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    pretrained_model_path="model/pretrained-com/KcBERT-Base",
    downstream_model_home=f"model/finetuned/kmou-ner-{now('%m%d')}",
    downstream_model_file="{epoch}, {val_loss:.3f}, {val_acc:.3f}",
    downstream_task_name="ner",
    downstream_data_home="data",
    downstream_data_name="kmou-ner",
    downstream_data_file="train.txt",
    max_seq_length=50,
    batch_size=100,
    accelerator="gpu",
    precision=16,
    strategy="single_device",
    devices=1,
    learning_rate=5e-5,
    epochs=1,
    seed=7,
).save_working_config()

cli.train_ner(config)
