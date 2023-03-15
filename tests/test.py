from chrislab.common.util import GpuProjectEnv

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0,1,2")
print(env.number_of_gpus)

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0")
print(env.number_of_gpus)
