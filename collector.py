import gym
import torch
import numpy as np
from torch.autograd import Variable

try:
    import gym_snake
except ImportError:
    print("You have not installed gym-snake! Do not try to use env_type=\'snake-v0\'")


class Collector():
    """
    This class handles the collection of data by interacting with the environments.
    """

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def cpu_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cpu()
        return tobj

    def __init__(self, reward_q, grid_size=[15,15], n_foods=1, unit_size=10, n_obs_stack=2, net=None, n_tsteps=15, gamma=0.99, env_type='snake-v0', preprocessor= lambda x: x):

        self.preprocess = preprocessor
        self.env_type = env_type
        self.env = gym.make(env_type)
        self.env.grid_size = grid_size
        self.env.n_foods = n_foods
        self.env.unit_size = unit_size
        self.action_space = self.env.action_space.n
        if env_type == 'Pong-v0':
            self.action_space = 2

        observation = self.env.reset()
        prepped_obs = self.preprocess(observation, env_type)
        self.obs_shape = observation.shape
        self.prepped_shape = prepped_obs.shape
        self.state_shape = [n_obs_stack*self.prepped_shape[0],*self.prepped_shape[1:]]
        self.state_bookmark = self.make_state(prepped_obs)

        self.gamma = gamma
        self.net = net
        self.n_tsteps = n_tsteps
        self.reward_q = reward_q

    def produce_data(self, data_q):
        """
        Used as the external call to get a rollout from each environment.

        Adds a tuple of data from a rollout to the process queue.
        data_q - multiprocessing.Queue that stores data to train the policy.
        """

        self.net.req_grads(False)
        self.net.train(mode=False)
        while True:
            data = self.rollout()
            data_q.put(data)



    def rollout(self):
        """
        Collects a rollout of n_tsteps in the given environment. The collected data
        are the states that were used to get the actions, the actions that
        were used to progress the environment, the rewards that were collected from
        the environment, and the done signals from the environment.

        Returns python lists of the relavent data.
        states - python list of all states collected in this rollout
        rewards - python list of float values collected from rolling out the environments
        dones - python list of booleans denoting the end of an episode
        actions - python list of integers denoting the actual selected actions in the
                    rollouts
        advantages - python list of floats denoting the td value error corresponding to
                    the equation r(t) + gamma*V(t+1) - V(t)
        """

        state = self.state_bookmark
        states, rewards, dones, actions, advantages = [], [], [], [], []
        for i in range(self.n_tsteps):
            tstate = self.cuda_if(torch.FloatTensor(state.copy()).unsqueeze(0))
            value, pi = self.net.forward(Variable(tstate))
            action = self.get_action(self.cpu_if(pi.data))

            obs, reward, done, info = self.env.step(action+2*(self.env_type == 'Pong-v0'))
            reset = done # Used to prevent reset in pong environment before actual done signal
            if reward != 0:
                if 'Pong-v0' == self.env_type: done = True
                self.reward_q.put(.99*self.reward_q.get() + .01*reward)

            value = value.squeeze().data[0]
            if i > 0:
                keep_val = (1-dones[-1]) # If the previous step was a done, then current value is not used
                advantage = rewards[-1] + self.gamma*value*keep_val - last_value
                advantages.append(advantage)
            last_value = value

            states.append(state), rewards.append(reward), dones.append(done), actions.append(action)
            state = self.next_state(self.env, state, obs, reset)

        self.state_bookmark = state
        if not done:
            tstate = self.cuda_if(torch.FloatTensor(state.copy()).unsqueeze(0))
            value, pi = self.net.forward(Variable(tstate))
            rewards[-1] = rewards[-1] + self.gamma*value.squeeze().data[0] # Bootstrapped value
            dones[-1] = True
        advantages.append(rewards[-1]-last_value)

        return states, rewards, dones, actions, advantages

    def get_action(self, pi):
        """
        Stochastically selects an action based on the action probabilities.

        pi - torch FloatTensor of the raw action prediction
        """

        action_ps = self.softmax(pi.numpy()).squeeze()
        action = np.random.choice(self.action_space, p=action_ps)
        return int(action)

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

    def next_state(self, env, prev_state, obs, reset):
        """
        Get the next state of the environment.

        env - environment of interest
        prev_state - ndarray of the state used in the most recent action
                    prediction
        obs - ndarray returned from the most recent step of the environment
        reset - boolean denoting the reset signal from the most recent step of the
                environment
        """

        if reset:
            obs = self.env.reset()
            prev_state = None
        prepped_obs = self.preprocess(obs, self.env_type)
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
