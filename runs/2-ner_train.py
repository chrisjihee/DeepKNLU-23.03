from nlpbook.arguments import *
from nlpbook.ner import cli

env = ProjectEnv(project="DeepKorNLU")
opt = env.running_file.stem.rsplit("-")[-1]
run_options = {
    "0": "pretrained-com/KLUE-BERT",
    "1": "pretrained-com/KLUE-RoBERTa",
    "2": "pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
    "3": "pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
    "4": "pretrained-com/KLUE-BERT",
    "5": "pretrained-com/KLUE-RoBERTa",
    "6": "pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
    "7": "pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
}
run_options["ner_train"] = run_options["0"]
assert opt in run_options, f"opt(={opt}) is not in {list(run_options.keys())}"

for learning_rate in [5e-5, 4e-5, 3e-5, 2e-5, 1e-5]:
    args = TrainerArguments(
        job=JobTimer(name=f"from-{Path(run_options[opt]).stem}"),
        env=ProjectEnv(project="DeepKorNLU", running_gpus=opt),
        data=DataOption(
            home="data",
            name="klue-ner",
            files=DataFiles(
                train="klue-ner-v1.1_train.jsonl",
                valid="klue-ner-v1.1_dev.jsonl",
            ),
            redownload=False,
            show_examples=0,
        ),
        model=ModelOption(
            pretrained=run_options[opt],
            finetuning_home="finetuning",
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
            num_keeping=5,
            keeping_by="max val_f1c",
            epochs=10,
            speed=learning_rate,
            seed=7,
        ),
    )
    job_name = args.job.name

    args.job.name = job_name + f"-using-fabric-lr={learning_rate}"
    with ArgumentsUsing(args) as args_file:
        cli.fabric_train(args_file)
