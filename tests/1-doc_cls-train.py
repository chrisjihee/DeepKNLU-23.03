from chrisbase.io import out_hr, out_table
from chrisbase.util import to_dataframe
from chrislab.common.util import GpuProjectEnv
from chrislab.ratsnlp import cli
from ratsnlp.nlpbook.classification import ClassificationTrainArguments

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0")
env.config_file.write_text(ClassificationTrainArguments(
    pretrained_model_name="pretrained/KcBERT-Base",
    downstream_corpus_root_dir="data",
    downstream_corpus_name="nsmc",
    downstream_model_dir="checkpoints/nsmc",
    monitor="max val_acc",
    learning_rate=5e-5,
    max_seq_length=128,
    batch_size=200,
    save_top_k=2,
    epochs=3,
    seed=7,
).to_json(ensure_ascii=False, indent=2, default=str))
if env.running_file.suffix == '.py':
    out_table(to_dataframe(env, columns=[env.__class__.__name__, "value"]))
    out_hr(c='-')

cli.train(env.config_file)
