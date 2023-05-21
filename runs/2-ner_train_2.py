from nlpbook.arguments import *
from nlpbook.ner import cli

args = TrainerArguments(
    env=ProjectEnv(
        project="DeepKorNLU",
        running_gpus="1",
    ),
    data=DataOption(
        home="data",
        name="klue-ner",
        files=DataFiles(
            train="klue-ner-v1.1_train.jsonl",
            valid="klue-ner-v1.1_dev.jsonl"
        ),
        caching=False,
        redownload=False,
    ),
    model=ModelOption(
        pretrained="model/pretrained-com/KLUE-BERT-Base",
        finetuning_home="model/finetuning_2",
        finetuning_name="{epoch:02d}, {step:04d}, {val_loss:.3f}, {val_acc:.3f}, {val_chr_f1:.3f}, {val_ent_f1:.3f}",
        max_seq_length=64,
    ),
    hardware=HardwareOption(
        accelerator="gpu",
        batch_size=100,
        precision=32,
    ),
    learning=LearningOption(
        condition="max val_chr_f1",
        log_steps=50,
        num_save=10,
        epochs=5,
        speed=5e-5,
        seed=7,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.train(args_file)
