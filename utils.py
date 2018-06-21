import torch
import numpy as np

def cuda_if(tobj):
    if torch.cuda.is_available():
        tobj = tobj.cuda()
    return tobj

def deque_maxmin(deq):
    max_val, min_val = deq[0],deq[0]
    for i in range(len(deq)):
        if deq[i] > max_val:
            max_val = deq[i]
        if deq[i] < min_val:
            min_val = deq[i]
    return max_val, min_val

def next_state(env, obs_deque, obs, reset, preprocess):
    """
    Get the next state of the environment.

    env - environment of interest
    obs_deq - deque of the past n observations
    obs - ndarray returned from the most recent step of the environment
    reset - boolean denoting the reset signal from the most recent step of the
            environment
    preprocess - function that handles preprocessing of raw observation
        type: function
    """

    if reset:
        obs = env.reset()
        prepped_obs = preprocess(obs)
        for i in range(obs_deque.maxlen-1):
            obs_deque.append(np.zeros(prepped_obs.shape))
    else:
        prepped_obs = preprocess(obs)
    obs_deque.append(prepped_obs)
    state = np.concatenate(obs_deque, axis=0)
    return state

def sample_action(pi):
    """
    Stochastically selects an action from the pi vectors.

    pi - torch FloatTensor that sums to 1 across the action space
        (i.e. a model output vector that has passed through a softmax)
        shape - (..., action_space)
    """
    pi = pi.cpu()
    rand_nums = torch.rand(*pi.shape[:-1])
    cumu_sum = torch.zeros(pi.shape[:-1])
    actions = -torch.ones(pi.shape[:-1])
    for i in range(pi.shape[-1]):
        cumu_sum += pi[...,i]
        actions[(cumu_sum >= rand_nums) & (actions < 0)] = i
    return actions


def discount(array, dones, discount_factor):
    """
    Dicounts the argued array following the bellman equation.

    array - array to be discounted
    dones - binary array denoting the end of an episode
    discount_factor - float between 0 and 1 used to discount the reward

    Returns the discounted array as a torch FloatTensor
    """
    running_sum = 0
    discounts = torch.zeros(len(array))
    for i in reversed(range(len(array))):
        if dones[i] == 1: running_sum = 0
        running_sum = array[i] + discount_factor*running_sum
        discounts[i] = running_sum
    return discounts

