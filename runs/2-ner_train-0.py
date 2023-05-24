from nlpbook.arguments import *
from nlpbook.ner import cli

args = TrainerArguments(
    job=JobTimer(name="from-KPF-BERT"),
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),
    data=DataOption(
        home="data",
        name="klue-ner-mini",
        files=DataFiles(
            train="klue-ner-v1.1_train.jsonl",
            valid="klue-ner-v1.1_dev.jsonl",
        ),
        redownload=False,
        show_examples=0,
    ),
    model=ModelOption(
        # pretrained="model/pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
        # pretrained="model/pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
        # pretrained="model/pretrained-com/KLUE-RoBERTa",
        pretrained="model/pretrained-com/KPF-BERT",
        finetuning_home="model/finetuning",
        finetuning_name="epoch={epoch:.1f}, val_acc={val_acc:.4f}",
        # finetuning_name="{epoch:02d}, {step:04d}, {chr_f1:.4f}, {ent_f1:.4f}",
        max_seq_length=64,
    ),
    hardware=HardwareOption(
        accelerator="gpu",
        batch_size=80,
        precision=16,
    ),
    learning=LearningOption(
        validating_on=0.5,
        num_keeping=3,
        keeping_by="max val_acc",
        epochs=3,
        speed=5e-5,
        seed=7,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.new_train(args_file)
