from nlpbook.arguments import *
from nlpbook.cls import cli

args = TrainerArguments(
    job=JobTimer(name="from-KPF-BERT"),
    env=ProjectEnv(project="DeepKNLU", running_gpus="0"),
    data=DataOption(
        home="data",
        name="nsmc",
        files=DataFiles(
            train="ratings_train.txt",
            valid="ratings_test.txt",
        ),
        redownload=False,
        show_examples=0,
    ),
    model=ModelOption(
        pretrained="pretrained-com/KPF-BERT",
        finetuning_home="finetuning",
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
        keeping_by="min val_loss",
        epochs=1,
        speed=5e-5,
        seed=7,
    ),
)
with ArgumentsUsing(args) as args_file:
    cli.train(args_file)
