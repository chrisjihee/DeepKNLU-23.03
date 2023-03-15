from chrisbase.util import to_dataframe
from chrislab.common.util import GpuProjectEnv

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0,1")
print(to_dataframe(env))
