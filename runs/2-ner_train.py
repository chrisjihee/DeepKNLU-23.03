from nlpbook.arguments import *
from nlpbook.ner import cli

args = TrainerArguments(
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0", debugging=False, tracing=False),
    data=DataOption(
        home="data",
        name="klue-ner-mini",
        files=DataFiles(
            train="klue-ner-v1.1_train.jsonl",
            valid="klue-ner-v1.1_dev.jsonl"
        ),
        caching=False,
        redownload=False,
    ),
    model=ModelOption(
        pretrained="model/pretrained-com/KcBERT-Base",
        finetuning_home="model/finetuning",
        finetuning_name="{epoch:02d}, {step:04d}, {train_loss:.4f}, {train_acc:.4f}, {val_loss:.4f}, {val_acc:.4f}",
        max_seq_length=7,
    ),
    hardware=HardwareOption(
        accelerator="gpu",
        batch_size=2,  # 100,
        precision=16,
    ),
    learning=LearningOption(
        condition="max val_acc",
        log_steps=3,  # 10,
        num_save=10,
        epochs=1,
        speed=5e-5,
        seed=7,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.train(args_file)
