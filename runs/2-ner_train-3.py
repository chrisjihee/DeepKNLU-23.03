from nlpbook.arguments import *
from nlpbook.ner import cli

args = TrainerArguments(
    env=ProjectEnv(
        project="DeepKorNLU",
        running_gpus="3",
    ),
    data=DataOption(
        home="data",
        name="klue-ner",
        files=DataFiles(
            train="klue-ner-v1.1_train.jsonl",
            valid="klue-ner-v1.1_dev.jsonl"
        ),
        redownload=False,
    ),
    model=ModelOption(
        pretrained="model/pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
        # pretrained="model/pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
        # pretrained="model/pretrained-com/KLUE-RoBERTa-Base",
        # pretrained="model/pretrained-com/KPF-BERT-Base",
        finetuning_home="model/finetuning",
        finetuning_name="{epoch:02d}, {step:04d}, {val_loss:.3f}, {val_acc:.3f}, {val_chr_f1:.3f}, {val_ent_f1:.3f}",
        max_seq_length=256,
    ),
    hardware=HardwareOption(
        accelerator="gpu",
        batch_size=80,
        precision=32,
    ),
    learning=LearningOption(
        condition="max val_chr_f1",
        log_steps=50,
        num_save=3,
        epochs=20,
        speed=5e-5,
        seed=7,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.train(args_file)
