from chrisbase.io import *
import pandas as pd

from chrisbase.util import to_dataframe
from nlpbook import TrainerArguments

ProjectEnv(project="DeepKNLU")

records = []
for indir in dirs("model/finetuning/klue-ner/train-from-*"):
    if (indir / "arguments-train.json").exists() and (indir / "metrics.csv").exists():
        args: TrainerArguments = TrainerArguments.from_json((indir / "arguments-train.json").read_text())
        data = pd.read_csv(indir / "metrics.csv")
        best = data[data['trained_rate'].notna()].sort_values(by='val_F1c', ascending=False).iloc[0].to_dict()
        records.append({
            "env_host": args.env.hostname,
            "data_name": args.data.name,
            "pretrained": Path(args.model.pretrained).name,
            "seq_length": args.model.max_seq_length,
            "batch_size": args.hardware.batch_size,
            "precision": args.hardware.precision,
            "trainer": args.job.name.split("-using-")[-1].split("-lr=")[0],
            "initial_lr": args.learning.speed,
            "runtime": args.job.elapsed,
            "epoch": best["trained_rate"] * 10,
            "val_loss": best["val_loss"],
            "val_F1c": best["val_F1c"],
            "val_F1e": best["val_F1e"],
        })
dataframe = to_dataframe(records)
print(dataframe)
dataframe.to_excel("runs/2-ner_train-summary.xlsx")
