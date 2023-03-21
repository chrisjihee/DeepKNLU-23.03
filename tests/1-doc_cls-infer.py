from chrislab.ratsnlp import cli
from chrisbase.io import out_table, out_hr
from chrisbase.util import to_dataframe
from chrislab.common.util import GpuProjectEnv
from ratsnlp.nlpbook.classification import ClassificationDeployArguments

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0")
out_table(to_dataframe(env, columns=[GpuProjectEnv.__name__, "value"]))
out_hr(c='-')
args = ClassificationDeployArguments(
    working_config_file=env.running_file.with_suffix('.json').name,
    pretrained_model_path="model/pretrained/KcBERT-Base",
    downstream_model_path="model/finetuned/nsmc",
    downstream_model_file=None,
    max_seq_length=128,
)
config = args.save_working_config()

assert config.exists(), f"No config file: {config}"
cli.deploy(config)
