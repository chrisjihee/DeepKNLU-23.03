from chrislab.common.util import GpuProjectEnv
from chrislab.ratsnlp import cli
from ratsnlp.nlpbook.classification import ClassificationTrainArguments

with GpuProjectEnv(project_name="DeepKorean", working_gpus="0") as env:
    args = ClassificationTrainArguments(
        pretrained_model_name="pretrained/KcBERT-Base",
        downstream_corpus_root_dir="data",
        downstream_corpus_name="nsmc",
        downstream_model_dir="checkpoints/nsmc",
        downstream_model_filename="{epoch}-{val_loss:.3f}-{val_acc:.3f}",
        monitor="max val_acc",
        learning_rate=5e-5,
        max_seq_length=128,
        cpu_workers=64,
        batch_size=200,
        save_top_k=3,
        epochs=3,
        seed=7,
    )
    env.config_file.write_text(args.to_json(ensure_ascii=False, indent=2, default=str))

cli.train(env.config_file)
