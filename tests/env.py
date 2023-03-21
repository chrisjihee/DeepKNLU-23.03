from chrisbase.io import environ_to_dataframe
from chrisbase.util import to_dataframe
from chrislab.common.util import BaseProjectEnv, GpuProjectEnv

env = BaseProjectEnv(project_name="DeepKorean")
print(to_dataframe(env))
print("=" * 80)

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0,1")
print(to_dataframe(env))
print("=" * 80)

print(environ_to_dataframe(max_value_len=60))
print("=" * 80)
