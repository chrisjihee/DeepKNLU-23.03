from chrisbase.io import BaseProjectEnv
from chrisbase.util import to_dataframe

env = BaseProjectEnv(project_name="DeepKorean")
print(to_dataframe(env))
print("=" * 80)

from chrisbase.util import to_dataframe
from chrislab.common.util import GpuProjectEnv

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0,1")
print(to_dataframe(env))
print("=" * 80)

from chrisbase.io import environ_to_dataframe

print(environ_to_dataframe(max_value_len=256))
print("=" * 80)
