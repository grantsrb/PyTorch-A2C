import sys
import torch
from torch.autograd import Variable
import gym
import numpy as np
import conv_model
import dense_model
from hyperparams import HyperParams
from utils import next_state, cuda_if, sample_action
import torch.nn.functional as F

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

preprocess = hyps['preprocess']
env_type = env_type
env = gym.make(env_type)
env.grid_size = grid_size
env.n_foods = n_foods
env.unit_size = unit_size
action_space = env.action_space.n
if env_type == 'Pong-v0':
    action_space = 3
    hyps['action_offset'] = 1
elif 'Breakout' in env_type:
    action_space = 4

obs = env.reset()
prepped_obs = preprocess(obs, env_type)
obs_shape = obs.shape
prepped_shape = prepped_obs.shape
state_shape = [n_frame_stack*prepped_shape[0],*prepped_shape[1:]]
state = make_state(prepped_obs)
net = hyps['model'](state_shape, action_space, bnorm=hyps['use_bnorm'])
net.load_state_dict(torch.load(file_name))
net.train(mode=False)
net.req_grads(False)

last_reset = 0
ep_reward = 0
counter = 0
while True:
    counter+=1
    value, pi = net.forward(Variable(torch.FloatTensor(state).unsqueeze(0)))
    pi = F.softmax(pi, dim=-1)
    action = int(get_action(pi.data).squeeze().item())
    obs, reward, done, info = env.step(action+hyps['action_offset'])
    ep_reward += reward
    env.render()
    if done:
        print("done", ep_reward)
        ep_reward=0
    state = next_state(env, state, obs, done, hyps['preprocess'], state_shape)
