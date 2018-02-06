import sys
from collector import Collector
from model import Model
import torch
from torch.autograd import Variable

file_name = str(sys.argv[1])
env_type = 'snake-v0'

# Environment Choices
grid_size = [15,15]
n_foods = 2
unit_size = 4
n_state_frames = 2 # number of observations to stack for a single environment state
action_space = 4


if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        str_arg = str(arg)
        if "grid_size=" in str_arg: grid_size= [int(str_arg[len('grid_size='):]),int(str_arg[len('grid_size='):])]
        if "n_foods=" in str_arg: n_foods= int(str_arg[len('n_foods='):])
        if "unit_size=" in str_arg: unit_size= int(str_arg[len('unit_size='):])
        if "n_state_frames=" in str_arg: n_state_frames= int(str_arg[len('n_state_frames='):])
        if "env_type=" in str_arg: env_type = str_arg[len('env_type='):]


print("file_name:", file_name)
print("n_state_frames:", n_state_frames)
print("grid_size:", grid_size)
print("n_foods:", n_foods)
print("unit_size:", unit_size)
print("env_type:", env_type)


collector = Collector(n_envs=1, grid_size=grid_size, n_foods=n_foods, unit_size=unit_size, n_state_frames=n_state_frames, net=None, n_tsteps=30, gamma=0, env_type=env_type)
net = Model(collector.state_shape, action_space)
collector.net = net
dummy = Variable(torch.ones(1,*collector.state_shape))
collector.net.calculate_grads(False)
collector.net.forward(dummy)
collector.net.train(mode=False)
collector.net.load_state_dict(torch.load(file_name))
print(collector.net.flat_shape)

while True:
    data = collector.get_data(True)
