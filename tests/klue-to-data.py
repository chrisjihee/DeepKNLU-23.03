from chrisbase.io import ProjectEnv
from chrisbase.util import to_dataframe

env = ProjectEnv(project="DeepKorNLU")
print(to_dataframe(env))

dataset_source = "data/klue/klue.py"
