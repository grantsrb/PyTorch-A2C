import sys
import torch
from torch.autograd import Variable
import gym
import numpy as np
from a2c.utils import next_state, cuda_if, sample_action
import torch.nn.functional as F
import multiprocessing as mp
from a2c.runner import StatsRunner
from a2c.models import *
import ml_utils.save_io as io
import matplotlib
matplotlib.use('TkAgg')

model_folder = sys.argv[1]
assert io.is_model_folder(model_folder), "Must argue a valid model folder!!"

checkpt = io.load_checkpoint(model_folder, use_best=True)
hyps = checkpt['hyps']
hyps['input_shape'] = hyps['state_shape']
hyps['output_space'] = hyps['action_size']
hyps['bnorm'] = hyps['use_bnorm']
for key,val in hyps.items():
    print(key + ":", val)

# Make Network
net = io.load_model(model_folder, globals(), use_best=True, hyps=hyps)
net.cuda()

hyps['n_test_eps'] = 50
hyps['render'] = True
runner = StatsRunner(hyps)

print("Avg reward:", runner.rollout(net))

