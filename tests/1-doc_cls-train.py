from chrislab.common.util import GpuProjectEnv
from chrislab.ratsnlp import cli
from ratsnlp.nlpbook.classification import ClassificationTrainArguments

with GpuProjectEnv(project_name="DeepKorean", working_gpus="4") as env:
    args = ClassificationTrainArguments(
        pretrained_model_path="model/pretrained/KcBERT-Base",
        downstream_model_path="model/finetuned/nsmc",
        downstream_model_file="{epoch}-{val_loss:.3f}-{val_acc:.3f}",
        downstream_data_home="data",
        downstream_data_name="nsmc",
        monitor="max val_acc",
        learning_rate=5e-5,
        max_seq_length=128,
        cpu_workers=12,
        batch_size=360,
        save_top_k=3,
        epochs=3,
        seed=7,
    )
    env.config_file.write_text(args.to_json(ensure_ascii=False, indent=2, default=str))

cli.train(env.config_file)
