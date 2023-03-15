import os
import sys
from pathlib import Path

import torch

from chrisbase.io import copy_dict, get_hostname, get_hostaddr, get_working_file, working_gpus, cwd, include_cuda_dir, dirs
from chrisbase.util import to_dataframe
from chrisdict import AttrDict

env = AttrDict()
env.hostname = get_hostname()
env.hostaddr = get_hostaddr()
env.python_path = Path(sys.executable)
env.project_name = "DeepKorean"
env.project_path = cwd([x for x in get_working_file().parents if env.project_name in x.name][0])
env.working_dir = Path.cwd()
env.working_file = get_working_file().relative_to(env.project_path)
env.working_gpus = working_gpus("0")
env.number_of_gpus = torch.cuda.device_count()
env.cuda_dir = include_cuda_dir()
env.torch_cuda_ver = torch.version.cuda

print(to_dataframe(copy_dict(dict(os.environ),
                             keys=[x for x in sorted(os.environ.keys()) if len(str(os.environ[x])) <= 100]),
                   columns=["os.environ", ""]))
print(to_dataframe(env, columns=["env", ""]))
