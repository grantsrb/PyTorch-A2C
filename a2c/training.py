import os
import sys
import gym
from a2c.logger import Logger
from a2c.runner import Runner
from a2c.updater import Updater
from a2c.utils import cuda_if, deque_maxmin
from a2c.models import *
import a2c.preprocessing as preprocessing
import torch
from torch.autograd import Variable
import numpy as np
import gc
import resource
import torch.multiprocessing as mp
import copy
import time
from collections import deque
from ml_utils.utils import try_key
from ml_utils.training import get_exp_num, record_session, get_save_folder
import ml_utils

def train(hyps, verbose=True): 
    """
    hyps - dictionary of required hyperparameters
        type: dict
    """
    # Hyperparam corrections
    if isinstance(try_key(hyps,"grid_size",None), int):
        hyps['grid_size'] = [hyps['grid_size'],hyps['grid_size']]

    # Preprocessor Type
    env_type = hyps['env_type'].lower()
    if "pong" in env_type:
        hyps['preprocess'] = preprocessing.pong_prep
    elif "breakout" in env_type:
        hyps['preprocess'] = preprocessing.breakout_prep
    elif "snake" in env_type:
        hyps['preprocess'] = preprocessing.snake_prep
    else:
        hyps['preprocess'] = preprocessing.atari_prep


    hyps['main_path'] = try_key(hyps, "main_path", "./")
    hyps['exp_num'] = get_exp_num(hyps['main_path'], hyps['exp_name'])
    hyps['save_folder'] = get_save_folder(hyps)
    save_folder = hyps['save_folder']
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    torch.manual_seed(hyps['seed'])
    np.random.seed(hyps['seed'])

    net_save_file = os.path.join(save_folder,"net.p")
    best_net_file = os.path.join(save_folder,"best_net.p")
    optim_save_file  = os.path.join(save_folder,"optim.p")
    log_file   = os.path.join(save_folder,"log.txt")

    if hyps['resume']: log = open(log_file, 'a')
    else: log = open(log_file, 'w')
    keys = sorted(list(hyps.keys()))
    for k in keys:
        log.write(k+":"+str(hyps[k])+"\n")

    # Miscellaneous Variable Prep
    logger = Logger()
    shared_len = hyps['n_tsteps']*hyps['n_rollouts']
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
    print("Num Samples Per Update:", shared_len)
    del env

    # Make Network
    hyps["action_size"] = action_size
    # TODO: make models compatible with **hyps
    net = globals()[hyps['model']](hyps['state_shape'], action_size,
                                   bnorm=hyps['use_bnorm'], **hyps)
    if try_key(hyps, 'resume', False):
        net.load_state_dict(torch.load(net_save_file))
    base_net = copy.deepcopy(net)
    net = cuda_if(net)
    net.share_memory()
    base_net = cuda_if(base_net)

    # Prepare Shared Variables
    zeros = torch.zeros(shared_len,*hyps['state_shape'])
    shared_data = {
            'states': cuda_if(zeros.share_memory_()),
            'deltas': cuda_if(torch.zeros(shared_len).share_memory_()),
            'rewards': cuda_if(torch.zeros(shared_len).share_memory_()),
            'actions': torch.zeros(shared_len).long().share_memory_(),
            'dones': cuda_if(torch.zeros(shared_len).share_memory_())}
    if net.is_recurrent:
        zeros = torch.zeros(shared_len,net.h_size)
        shared_data['h_states'] = cuda_if(zeros.share_memory_())
    n_rollouts = hyps['n_rollouts']
    gate_q = mp.Queue(n_rollouts)
    stop_q = mp.Queue(n_rollouts)
    reward_q = mp.Queue(1)
    reward_q.put(-1)

    # Make Runners
    runners = []
    for i in range(hyps['n_envs']):
        runner = Runner(shared_data, hyps, gate_q, stop_q, reward_q)
        runners.append(runner)

    # Start Data Collection
    print("Making New Processes")
    procs = []
    for i in range(len(runners)):
        proc = mp.Process(target=runners[i].run, args=(net,))
        procs.append(proc)
        proc.start()
        print(i, "/", len(runners), end='\r')
    for i in range(n_rollouts):
        gate_q.put(i)

    # Make Updater
    updater = Updater(base_net, hyps)
    if hyps['resume']:
        updater.optim.load_state_dict(torch.load(optim_save_file))
    updater.optim.zero_grad()
    updater.net.train(mode=True)
    updater.net.req_grads(True)

    # Prepare Decay Precursors
    entr_coef_diff = hyps['entr_coef'] - hyps['entr_coef_low']
    lr_diff = hyps['lr'] - hyps['lr_low']
    gamma_diff = hyps['gamma_high'] - hyps['gamma']

    # Training Loop
    past_rews = deque([0]*hyps['n_past_rews'])
    last_avg_rew = 0
    best_avg_rew = -100
    epoch = 0
    T = 0
    while T < hyps['max_tsteps']:
        basetime = time.time()
        epoch += 1
        stats_string = ""

        # Collect data
        for i in range(n_rollouts):
            stop_q.get()
        T += shared_len
        s = "Epoch {} - T: {} -- {}".format(epoch,T,hyps['save_folder'])
        print(s)
        stats_string += s + "\n"

        # Reward Stats
        avg_reward = reward_q.get()
        reward_q.put(avg_reward)
        last_avg_rew = avg_reward
        if avg_reward > best_avg_rew:
            best_avg_rew = avg_reward
            updater.save_model(best_net_file, None)

        # Calculate the Loss and Update nets
        updater.update_model(shared_data)
        # update all collector nets
        net.load_state_dict(updater.net.state_dict()) 
        
        # Resume Data Collection
        for i in range(n_rollouts):
            gate_q.put(i)

        # Decay HyperParameters
        if hyps['decay_lr']:
            decay_factor = max((1-T/(hyps['max_tsteps'])), 0)
            new_lr = decay_factor*lr_diff + hyps['lr_low']
            updater.new_lr(new_lr)
            s = "New lr: "+str(new_lr)
            print(s)
            stats_string += s + "\n"
        if hyps['decay_entr']:
            decay_factor = max((1-T/(hyps['max_tsteps'])), 0)
            updater.entr_coef = entr_coef_diff*decay_factor
            updater.entr_coef += hyps['entr_coef_low']
            s = "New Entr: " + str(updater.entr_coef)
            print(s)
            stats_string += s + "\n"

        # Periodically save model
        if epoch % 10 == 0:
            updater.save_model(net_save_file, optim_save_file)

        # Print Epoch Data
        past_rews.popleft()
        past_rews.append(avg_reward)
        max_rew, min_rew = deque_maxmin(past_rews)
        rew_avg, rew_std = np.mean(past_rews), np.std(past_rews)
        updater.print_statistics()
        avg_action = shared_data['actions'].float().mean().item()
        s="Grad Norm: {:.5f} – Avg Action: {:.5f} - Best AvgRew: {:.5f}"
        s = s.format(float(updater.norm), avg_action, best_avg_rew)
        stats_string += s + "\n"
        s = "Avg Rew: " + str(avg_reward)
        stats_string += s + "\n"
        s = "Past "+str(hyps['n_past_rews'])+" Rews – High: {:.5f}"
        s += " - Low: {:.5f} - Avg: {:.5f} - StD: {:.5f}"
        stats_string += s.format(max_rew, min_rew, rew_avg, rew_std)+"\n"
        updater.log_statistics(log, T, avg_reward, avg_action,
                                                   best_avg_rew)
        updater.info['AvgRew'] = avg_reward
        logger.append(updater.info, x_val=T)

        # Check for memory leaks
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        s = "Time: " + str(time.time()-basetime)
        stats_string += s + "\n"
        print(stats_string)
        log.write(stats_string + "\n")
        if 'hyp_search_count' in hyps and hyps['hyp_search_count'] > 0\
                                         and hyps['search_id'] != None:
            print("Search:", hyps['search_id'], "/",
                            hyps['hyp_search_count'])
        print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))

    logger.make_plots(save_folder+hyps['exp_name'])
    log.write("\nBestRew:"+str(best_avg_rew))
    log.close()

    # Close processes
    for p in procs:
        p.terminate()

    return best_avg_rew
