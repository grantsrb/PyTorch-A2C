import sys
import torch
from torch.autograd import Variable
import gym
import numpy as np
from hyperparams import HyperParams
from utils import next_state, cuda_if, sample_action
import torch.nn.functional as F
import multiprocessing as mp
from runner import Runner

hp = HyperParams()
hyps = hp.hyps
file_name = hyps['exp_name'] + "_best.p"
for arg in sys.argv[1:]:
    if "file_name" in str(arg) or "=" not in str(arg) or ".p" in str(arg):
        file_name = arg
        break

print("file_name:", file_name)
print("n_frame_stack:", hyps['n_frame_stack'])
print("grid_size:", hyps['grid_size'])
print("n_foods:", hyps['n_foods'])
print("unit_size:", hyps['unit_size'])
print("env_type:", hyps['env_type'])
print("model_type:", hyps['model_type'])
print("preprocessor:", hyps['preprocess'])

hyps['render'] = True
preprocess = hyps['preprocess']
env_type = hyps['env_type']
env = gym.make(env_type)
action_space = env.action_space.n
if env_type == 'Pong-v0':
    action_space = 3
    hyps['action_offset'] = 1
elif 'Breakout' in env_type:
    action_space = 4

# Miscellaneous Variable Prep
env = gym.make(hyps['env_type'])
obs = env.reset()
prepped = hyps['preprocess'](obs)
hyps['state_shape'] = [hyps['n_frame_stack']] + [*prepped.shape[1:]]
if hyps['env_type'] == "Pong-v0":
    action_size = 3
else:
    action_size = env.action_space.n
hyps['action_shift'] = (4-action_size)*(hyps['env_type']=="Pong-v0") 
print("Obs Shape:,",obs.shape)
print("Prep Shape:,",prepped.shape)
print("State Shape:,",hyps['state_shape'])
del env

# Make Network
net = hyps['model'](hyps['state_shape'], action_size, h_size=hyps['h_size'], bnorm=hyps['use_bnorm'])
net.load_state_dict(torch.load(file_name))
net = cuda_if(net)

# Prepare Shared Variables
shared_len = hyps['n_tsteps']
shared_data = {'states': cuda_if(torch.zeros(shared_len, *hyps['state_shape']).share_memory_()),
        'deltas': cuda_if(torch.zeros(shared_len).share_memory_()),
        'rewards': cuda_if(torch.zeros(shared_len).share_memory_()),
        'actions': torch.zeros(shared_len).long().share_memory_(),
        'dones': cuda_if(torch.zeros(shared_len).share_memory_())}
if net.is_recurrent:
    shared_data['h_states'] = cuda_if(torch.zeros(shared_len, net.h_size).share_memory_())
gate_q = mp.Queue(1)
stop_q = mp.Queue(1)
reward_q = mp.Queue(1)
reward_q.put(-1)

# Make Runner
runner = Runner(shared_data, hyps, gate_q, stop_q, reward_q)

# Start Runner
proc = mp.Process(target=runner.run, args=(net,))
proc.start()
gate_q.put(0)

while True:
    stop_q.get()
    gate_q.put(0)
