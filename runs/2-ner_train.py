from nlpbook.arguments import *
from nlpbook.ner import cli

args = TrainerArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    model=ModelArgs(
        data_file=DataFiles(train="train.txt", valid="valid.txt"),
        data_name="kmou-ner",
        data_home="data",
        data_caching=False,
        finetuned_name="{epoch:02d}, {step:04d}, {train_loss:.4f}, {train_acc:.4f}, {val_loss:.4f}, {val_acc:.4f}",
        finetuned_home="model/finetuned",
        pretrained_name="model/pretrained-com/KcBERT-Base",
        max_seq_length=50,
    ),
    hardware=HardwareArgs(
        batch_size=100,
        accelerator="gpu",
        precision=16,
    ),
    training=TrainingArgs(
        learning_rate=5e-5,
        save_top_k=100,
        log_steps=10,
        monitor="min val_loss",
        epochs=1,
        seed=7,
    ),
)
with UsingArguments(args) as args_file:
    cli.train(args_file)
