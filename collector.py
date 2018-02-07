import gym
import gym_snake
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Collector():
    """
    This class handles the collection of data by interacting with the environments.
    """

    if torch.cuda.is_available():
        torch.FloatTensor = torch.cuda.FloatTensor
        torch.LongTensor = torch.cuda.LongTensor

    def __init__(self, n_envs=1, grid_size=[15,15], n_foods=1, unit_size=10, n_state_frames=3, net=None, n_tsteps=15, gamma=0.99, env_type='snake-v0', preprocessor= lambda x: x):

        self.preprocess = preprocessor
        self.n_envs = n_envs
        self.envs = [gym.make(env_type) for env in range(n_envs)]
        for i in range(n_envs):
            self.envs[i].grid_size = grid_size
            self.envs[i].n_foods = n_foods
            self.envs[i].unit_size = unit_size

        observations = [self.envs[i].reset() for i in range(n_envs)]
        self.PREP_MEAN = np.mean(observations[0])
        prepped_observations = [self.preprocess(obs) for obs in observations]
        self.prepped_shape = prepped_observations[0].shape
        prepped_obs = prepped_observations[0] # Moves channels to first dimension
        self.state_shape = [n_state_frames*prepped_obs.shape[0],*prepped_obs.shape[1:]]
        self.state_bookmarks = [self.make_state(obs) for obs in prepped_observations]

        self.gamma = gamma
        self.net = net
        self.n_tsteps = n_tsteps
        self.T = 0
        self.avg_reward = 0



    def get_data(self, render=False):
        """
        Used as the external call to get a rollout from each environment.

        Returns python lists of the relavent data.
        ep_states - ndarray with dimensions of [num_samples, *state_shape]
        ep_rewards - python list of float values collected from rolling out the environments
        ep_dones - python list of booleans denoting the end of an episode
        ep_actions - python list of integers denoting the actual selected actions in the
                    rollouts
        ep_advantages - python list of floats denoting the td value error corresponding to
                    the equation r(t) + gamma*V(t+1) - V(t)
        """

        self.net.req_grads(False)
        self.net.train(mode=False)
        ep_states, ep_rewards, ep_dones, ep_actions, ep_advantages = [], [], [], [], []
        for i in range(self.n_envs):
            ep_states, ep_rewards, ep_dones, ep_actions, ep_advantages = self.rollout(i, ep_states, ep_rewards, ep_dones, ep_actions, ep_advantages, render and i==0)
        self.net.req_grads(True)
        self.net.train(mode=True)
        return np.asarray(ep_states,dtype=np.float32), ep_rewards, ep_dones, ep_actions, ep_advantages

    def rollout(self, env_idx, states, rewards, dones, actions, advantages, render=False):
        """
        Collects a rollout of n_tsteps in the given environment. The collected data
        are the states that were used to get the actions, the actions that
        were used to progress the environment, the rewards that were collected from
        the environment, and the done signals from the environment.

        env_idx - integer index of the environment to be rolled out
        states - python list of states accumulated during the current epoch
        rewards - python list of rewards accumulated during the current epoch
        dones - python list of dones accumulated during the current epoch
        actions - python list of actions accumulated during the current epoch
        advantages - python list of advantages accumulated during the current epoch

        Returns python lists of the relavent data.
        states - python list of all states collected in this rollout
        rewards - python list of float values collected from rolling out the environments
        dones - python list of booleans denoting the end of an episode
        actions - python list of integers denoting the actual selected actions in the
                    rollouts
        advantages - python list of floats denoting the td value error corresponding to
                    the equation r(t) + gamma*V(t+1) - V(t)
        """

        state = self.state_bookmarks[env_idx]
        for i in range(self.n_tsteps):
            if render:
                self.envs[env_idx].render(mode='human')
            self.T += 1

            value, pi = self.net.forward(Variable(torch.FloatTensor(state).unsqueeze(0)))
            action = self.get_action(pi.data)
            obs, reward, done, info = self.envs[env_idx].step(action)
            if reward is not 0: self.avg_reward = .99*self.avg_reward + 0.01*reward

            value = value.squeeze().data[0]
            if i > 0:
                advantage = self.temporal_difference(rewards[-1], value*(1-dones[-1]), last_value)
                last_value = value
                advantages.append(advantage)
            else:
                last_value = value

            states.append(state), rewards.append(reward), dones.append(done), actions.append(action)
            state = self.next_state(env_idx, state, obs, done)

        self.state_bookmarks[env_idx] = state
        if not done:
            value, pi = self.net.forward(Variable(torch.FloatTensor(state).unsqueeze(0)))
            value = value.squeeze().data[0]
            rewards[-1] = reward + self.gamma*value # Bootstrapped value
            advantage = self.temporal_difference(rewards[-1], 0, last_value)
            advantages.append(advantage)
            dones[-1] = True
        else:
            advantages.append(rewards[-1]-last_value)

        return states, rewards, dones, actions, advantages

    def get_action(self, pi):
        """
        Stochastically selects an action based on the action probabilities.

        pi - torch FloatTensor of the raw action prediction
        """

        action_ps = self.softmax(pi.numpy()).squeeze()
        action = np.random.choice(self.net.output_space, p=action_ps)
        return action

    def make_state(self, prepped_obs, prev_state=None):
        """
        Combines the new, prepprocessed observation with the appropriate parts of the previous
        state to make a new state that is ready to be passed through the network. If prev_state
        is None, the state is filled with zeros outside of the new observation.

        prepped_obs - torch FloatTensor of peprocessed observation
        prev_state - torch FloatTensor of previous environment state
        """

        if prev_state is None:
            prev_state = np.zeros(self.state_shape, dtype=np.float32)

        next_state = np.concatenate([prepped_obs, prev_state[:-prepped_obs.shape[0]]], axis=0)
        return next_state

    def next_state(self, env_idx, prev_state, obs, done):
        """
        Get the next state of the environment corresponding to the env_idx.

        env_idx - integer index denoting environment of interest
        prev_state - ndarray of the state used in the most recent action
                    prediction
        obs - ndarray returned from the most recent step of the environment
        done - boolean denoting the done signal from the most recent step of the
                environment
        """

        if done:
            obs = self.envs[env_idx].reset()
            prev_state = None
        prepped_obs = self.preprocess(obs)
        state = self.make_state(prepped_obs, prev_state)
        return state

    def preprocess(self, pic):
        """
        Each raw observation from the environment is run through this function.
        Put anything sort of preprocessing into this function.
        This function is set in the intializer.

        pic - ndarray of an observation from the environment [H,W,C]
        """
        pass

    def softmax(self, X, theta=1.0, axis=-1):
        """
        * Inspired by https://nolanbconaway.github.io/blog/2017/softmax-numpy *

        Computes the softmax of each element along an axis of X.

        X - ndarray of at least 2 dimensions
        theta - float used as a multiplier prior to exponentiation
        axis - axis to compute values along

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """

        X = X * float(theta)
        X = X - np.expand_dims(np.max(X, axis = axis), axis)
        X = np.exp(X)
        ax_sum = np.expand_dims(np.sum(X, axis = axis), axis)
        p = X / ax_sum
        return p

    def temporal_difference(self, actual, next_pred, current_pred):
        """
        Calculates the temporal difference of two predictions.

        actual - float referring to the actual thing (often the reward) collected at time t.
        next_pred - float referring to the Q or val prediction of the next state
        current_pred - float refferring to the current Q or val prediction
        """

        return actual + self.gamma*next_pred - current_pred
