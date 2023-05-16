from chrisbase.io import ProjectEnv
from nlpbook.arguments import CommandConfig, NLUTrainerArguments, DataFiles
from nlpbook.ner import cli

args = NLUTrainerArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    downstream_data_file=DataFiles(train="train.txt", valid="valid.txt"),
    downstream_data_name="kmou-ner",
    downstream_data_home="data",
    downstream_data_caching=False,
    downstream_model_file="{epoch:02d}, {step:04d}, {train_loss:.4f}, {train_acc:.4f}, {val_loss:.4f}, {val_acc:.4f}",
    downstream_model_home="model/finetuned",
    pretrained_model_path="model/pretrained-com/KcBERT-Base",
    max_seq_length=50,
    batch_size=100,
    accelerator="gpu",
    precision=16,
    learning_rate=5e-5,
    save_top_k=100,
    log_steps=10,
    monitor="min val_loss",
    epochs=1,
    seed=7,
)
with CommandConfig(args) as config:
    cli.train(config)
