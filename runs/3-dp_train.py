from nlpbook.arguments import *
from nlpbook.ner import cli

env = ProjectEnv(project="DeepKorNLU")
opt = env.running_file.stem.rsplit("-")[-1]
run_options = {
    "0": "model/pretrained-com/KLUE-BERT",
    "1": "model/pretrained-com/KLUE-RoBERTa",
    "2": "model/pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
    "3": "model/pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
    "4": "model/pretrained-com/KLUE-BERT",
    "5": "model/pretrained-com/KLUE-RoBERTa",
    "6": "model/pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
    "7": "model/pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
}
run_options["dp_train"] = run_options["0"]
assert opt in run_options, f"opt(={opt}) is not in {list(run_options.keys())}"

for learning_rate in [5e-5]:  # , 4e-5, 3e-5, 2e-5, 1e-5]:
    args = TrainerArguments(
        job=JobTimer(name=f"from-{Path(run_options[opt]).stem}"),
        env=ProjectEnv(project="DeepKorNLU", running_gpus=opt),
        data=DataOption(
            home="data",
            name="klue-dp-mini",
            files=DataFiles(
                train="klue-dp-v1.1_train.tsv",
                valid="klue-dp-v1.1_dev.tsv",
            ),
            redownload=False,
            show_examples=0,
        ),
        model=ModelOption(
            pretrained=run_options[opt],
            finetuning_home="model/finetuning",
            finetuning_name="epoch={epoch:.1f}, trained_rate={trained_rate:.2f}, f1c={val_f1c:05.2f}, f1e={val_f1e:05.2f}",
            max_seq_length=128,
        ),
        hardware=HardwareOption(
            accelerator="gpu",
            batch_size=100,
            precision="16-mixed",
        ),
        learning=LearningOption(
            validating_fmt="loss={val_loss:06.4f}, f1c={val_f1c:05.2f}, f1e={val_f1e:05.2f}",
            validating_on=1 / 10,
            num_keeping=2,
            keeping_by="max val_f1c",
            epochs=2,
            speed=learning_rate,
            seed=7,
        ),
    )
    job_name = args.job.name

    args.job.name = job_name + f"-using-fabric1-lr={learning_rate}"
    with ArgumentsUsing(args) as args_file:
        cli.fabric_train1(args_file)

    args.job.name = job_name + f"-using-fabric2-lr={learning_rate}"
    with ArgumentsUsing(args) as args_file:
        cli.fabric_train2(args_file)
