# Please, unset LD_LIBRARY_PATH before run this script
# from multiprocessing import freeze_support

import torch.cuda

from nlpbook.arguments import *
from nlpbook.ner import cli

if __name__ == '__main__':
    env = ProjectEnv(project="DeepKNLU")
    opt = env.running_file.stem.rsplit("-")[-1]
    use_gpu = torch.cuda.is_available()
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

    # freeze_support()
    for learning_rate in [5e-5, 4e-5, 3e-5, 2e-5, 1e-5]:
        args = TrainerArguments(
            job=JobTimer(name=f"from-{Path(run_options[opt]).stem}"),
            env=ProjectEnv(project="DeepKNLU", running_gpus=opt if use_gpu else None),
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
                finetuning_name="epoch={epoch:.1f}, tr={trained_rate:.2f}, F1c={val_F1c:05.2f}, F1e={val_F1e:05.2f}",
                max_seq_length=256,
            ),
            hardware=HardwareOption(
                accelerator="gpu" if use_gpu else "cpu",
                precision="16-mixed" if use_gpu else "bf16-mixed",
                batch_size=100,
            ),
            learning=LearningOption(
                validating_fmt="loss={val_loss:06.4f}, F1c={val_F1c:05.2f}, F1e={val_F1e:05.2f}",
                validating_on=0.1,
                num_keeping=2,
                keeping_by="max val_F1c",
                epochs=10,
                speed=learning_rate,
                seed=7,
            ),
        )
        job_name = args.job.name

        args.job.name = job_name + f"-using-fabric-lr={learning_rate}"
        with ArgumentsUsing(args) as args_file:
            cli.fabric_train(args_file)
