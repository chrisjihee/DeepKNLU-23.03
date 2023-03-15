import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from chrisbase.io import cwd, first_or, copy_dict, get_hostname, get_hostaddr, working_file, working_gpus, include_cuda_bin_dir
from chrisbase.util import to_dataframe


@dataclass
class EnvArgs:
    hostname: str = field(init=False)
    hostaddr: str = field(init=False)
    python_path: Path = field(init=False)
    project_name: str = field()
    project_path: Path = field(init=False)
    working_dir: Path = field(init=False)
    working_file: Path = field(init=False)
    working_gpus: str = field(default="0")
    number_of_gpus: int = field(init=False, default=0)
    cuda_home_dir: Path = field(init=False)
    torch_cuda_ver: str = field(init=False)

    def __post_init__(self):
        self.hostname = get_hostname()
        self.hostaddr = get_hostaddr()
        self.python_path = Path(sys.executable)
        self.project_path = cwd(first_or([x for x in working_file().parents if self.project_name and x.name.startswith(self.project_name)]))
        self.working_dir = Path.cwd()
        self.working_file = working_file().relative_to(self.project_path)
        self.working_gpus = working_gpus(self.working_gpus)
        self.number_of_gpus = torch.cuda.device_count()
        self.cuda_home_dir = include_cuda_bin_dir()
        self.torch_cuda_ver = torch.version.cuda


env = EnvArgs(project_name="DeepKorean", working_gpus="0,1")

print(to_dataframe(copy_dict(dict(os.environ),
                             keys=[x for x in sorted(os.environ.keys()) if len(str(os.environ[x])) <= 100]),
                   columns=["os.environ", ""]))
print(to_dataframe(asdict(env), columns=["env", ""]))
