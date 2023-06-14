# Please, unset LD_LIBRARY_PATH before run this script
from multiprocessing import freeze_support

import torch.cuda

from nlpbook.arguments import *
from nlpbook.dp import cli

env = ProjectEnv(project="DeepKNLU")
opt = env.running_file.stem.rsplit("-")[-1]
run_options = {
    "0": "pretrained-com/KLUE-RoBERTa",
    "1": "pretrained-com/KLUE-BERT",
    "2": "pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
    "3": "pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
    "4": "pretrained-com/KLUE-RoBERTa",
    "5": "pretrained-com/KLUE-BERT",
    "6": "pretrained-pro/ETRI-RoBERTa-Base-bbpe23.03",
    "7": "pretrained-pro/ETRI-RoBERTa-Base-bbpe22.07",
}
if opt not in run_options:
    opt = list(run_options.keys())[0]
assert opt in run_options, f"opt(={opt}) is not in {list(run_options.keys())}"

if __name__ == '__main__':
    freeze_support()
    for learning_rate in [5e-5]:  # , 4e-5, 3e-5, 2e-5, 1e-5]:
        args = TrainerArguments(
            job=JobTimer(name=f"from-{Path(run_options[opt]).stem}"),
            env=ProjectEnv(project="DeepKNLU",
                           running_gpus=opt if torch.cuda.is_available() else None,
                           # off_debugging=True,
                           ),
            data=DataOption(
                home="data",
                name="klue-dp-mini",
                files=DataFiles(
                    train="klue-dp-v1.1_train.tsv",
                    valid="klue-dp-v1.1_dev.tsv",
                ),
                redownload=False,
                show_examples=3,
            ),
            model=ModelOption(
                pretrained=run_options[opt],
                finetuning_home="finetuning",
                finetuning_name="epoch={epoch:.1f}, trained_rate={trained_rate:.2f}, f1c={val_f1c:05.2f}, f1e={val_f1e:05.2f}",
                max_seq_length=128,
            ),
            hardware=HardwareOption(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                batch_size=32,
                precision="16-mixed" if torch.cuda.is_available() else "bf16-mixed",
            ),
            learning=LearningOption(
                validating_fmt="loss={val_loss:06.4f}",
                # validating_fmt="loss={val_loss:06.4f}, f1c={val_f1c:05.2f}, f1e={val_f1e:05.2f}",
                validating_on=1 / 10,
                num_keeping=2,
                keeping_by="max val_f1c",
                epochs=2,
                speed=learning_rate,
                seed=42,
            ),
        )
        job_name = args.job.name

        args.job.name = job_name + f"-using-fabric1-lr={learning_rate}"
        with ArgumentsUsing(args) as args_file:
            cli.fabric_train(args_file)
