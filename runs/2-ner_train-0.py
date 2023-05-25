from nlpbook.arguments import *
from nlpbook.ner import cli

env = ProjectEnv(project="DeepKorNLU")
opt = env.running_file.stem.rsplit("-")[-1]
run_options = {
    "0": "model/pretrained-com/KLUE-BERT",
    "1": "model/pretrained-com/KLUE-RoBERTa",
    "2": "model/pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
    "3": "model/pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
}
assert opt in run_options

args = TrainerArguments(
    job=JobTimer(name=f"from-{Path(run_options[opt]).stem}"),
    env=ProjectEnv(project="DeepKorNLU", running_gpus=opt),
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
        pretrained=run_options[opt],
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
)
job_name = args.job.name

args.job.name = job_name + "-using-fabric_train1"
with ArgumentsUsing(args) as args_file:
    cli.fabric_train1(args_file)

args.job.name = job_name + "-using-fabric_train2"
with ArgumentsUsing(args) as args_file:
    cli.fabric_train2(args_file)
