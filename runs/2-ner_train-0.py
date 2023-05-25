from nlpbook.arguments import *
from nlpbook.ner import cli

args = TrainerArguments(
    # job=JobTimer(name="from-KPF-BERT"),
    env=ProjectEnv(project="DeepKorNLU", running_gpus="0"),  # running_gpus by filename
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
        finetuning_name="epoch={epoch:.1f}, f1c={val_f1c:05.2f}, f1e={val_f1e:05.2f}",
        max_seq_length=64,
    ),
    hardware=HardwareOption(
        accelerator="gpu",
        batch_size=50,
        precision="16-mixed",
    ),
    learning=LearningOption(
        validating_fmt="loss={val_loss:06.4f}, f1c={val_f1c:05.2f}, f1e={val_f1e:05.2f}",
        validating_on=1 / 10,
        num_keeping=5,
        keeping_by="max val_f1c",
        epochs=3,
        speed=5e-5,
        seed=7,
    ),
    job=JobTimer(name="fabric_train2"),
)
with ArgumentsUsing(args) as args_file:
    cli.fabric_train2(args_file)
