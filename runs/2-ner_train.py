from nlpbook.arguments import *
from nlpbook.ner import cli

args = TrainerArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    data=DataOption(
        home="data",
        name="kmou-ner",
        files=DataFiles(train="train.jsonl", valid="valid.jsonl"),
        caching=False,
    ),
    model=ModelOption(
        pretrained="model/pretrained-com/KcBERT-Base",
        finetuning_home="model/finetuning",
        finetuning_name="{epoch:02d}, {step:04d}, {train_loss:.4f}, {train_acc:.4f}, {val_loss:.4f}, {val_acc:.4f}",
        max_seq_length=50,
    ),
    hardware=HardwareOption(
        accelerator="gpu",
        batch_size=100,
        precision=16,
    ),
    learning=LearningOption(
        condition="min val_loss",
        log_steps=10,
        num_save=100,
        epochs=1,
        speed=5e-5,
        seed=7,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.train(args_file)
