from nlpbook.arguments import *
from nlpbook.ner import cli

args = TrainerArguments(
    env=ProjectEnv(
        project="DeepKorNLU",
        running_gpus="0",
        on_debugging=True,
        # on_tracing=True,
    ),
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
        max_seq_length=50,
        # max_seq_length=7,
    ),
    hardware=HardwareOption(
        accelerator="gpu",
        # batch_size=100,
        batch_size=20,
        precision=16,
    ),
    learning=LearningOption(
        condition="min val_loss",
        log_steps=50,
        # log_steps=3,
        num_save=10,
        epochs=1,
        speed=5e-5,
        seed=7,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.train(args_file)
