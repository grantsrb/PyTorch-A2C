import sys
from sequential_collector import Collector
import torch
from torch.autograd import Variable

# Change your policy file here!
import dense_model as model
print("Using dense_model as policy file.")

file_name = str(sys.argv[1])
env_type = 'snake-v0'

# Environment Choices
grid_size = [15,15]
n_foods = 2
unit_size = 4
n_obs_stack = 2 # number of observations to stack for a single environment state


if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        str_arg = str(arg)
        if "grid_size=" in str_arg: grid_size= [int(str_arg[len('grid_size='):]),int(str_arg[len('grid_size='):])]
        if "n_foods=" in str_arg: n_foods= int(str_arg[len('n_foods='):])
        if "unit_size=" in str_arg: unit_size= int(str_arg[len('unit_size='):])
        if "n_obs_stack=" in str_arg: n_obs_stack= int(str_arg[len('n_obs_stack='):])
        if "env_type=" in str_arg: env_type = str_arg[len('env_type='):]

print("file_name:", file_name)
print("n_obs_stack:", n_obs_stack)
print("grid_size:", grid_size)
print("n_foods:", n_foods)
print("unit_size:", unit_size)
print("env_type:", env_type)


collector = Collector(n_envs=1, grid_size=grid_size, n_foods=n_foods, unit_size=unit_size, n_obs_stack=n_obs_stack, net=None, n_tsteps=30, gamma=0, env_type=env_type, preprocessor=model.Model.preprocess)
net = model.Model(collector.state_shape, collector.action_space, env_type=env_type)
collector.net = net
dummy = Variable(torch.ones(2,*collector.state_shape))
collector.net.forward(dummy)
collector.net.load_state_dict(torch.load(file_name))
collector.net.train(mode=False)
collector.net.req_grads(False)

while True:
    data = collector.get_data(render=True)
