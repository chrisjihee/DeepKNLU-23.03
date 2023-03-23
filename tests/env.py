from chrisbase.io import environ_to_dataframe
from chrisbase.util import to_dataframe
from chrislab.common.util import BaseProjectEnv

env = BaseProjectEnv(project_name="DeepKorean")
print(to_dataframe(env))
print("=" * 80)

print(environ_to_dataframe(max_value_len=60))
print("=" * 80)
