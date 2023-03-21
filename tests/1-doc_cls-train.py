from chrisbase.io import out_table, out_hr
from chrisbase.util import to_dataframe
from chrislab.common.util import GpuProjectEnv
from chrislab.ratsnlp import cli
from ratsnlp.nlpbook.classification import ClassificationTrainArguments

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0")
out_table(to_dataframe(env, columns=[GpuProjectEnv.__name__, "value"]))
out_hr(c='-')
args = ClassificationTrainArguments(
    working_config_file=env.running_file.with_suffix('.json').name,
    pretrained_model_path="model/pretrained/KcBERT-Base",
    downstream_model_path="model/finetuned/nsmc",
    downstream_model_file="{epoch}-{val_loss:.3f}-{val_acc:.3f}",
    downstream_data_home="data",
    downstream_data_name="nsmc",
    monitor="max val_acc",
    learning_rate=5e-5,
    max_seq_length=128,
    cpu_workers=24,
    batch_size=360,
    save_top_k=3,
    epochs=3,
    seed=7,
)
config = args.save_working_config()

assert config.exists(), f"No config file: {config}"
cli.train(config)
