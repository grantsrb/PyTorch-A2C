import time
from ml_utils.utils import try_key
from a2c.utils import next_state, sample_action, cuda_if
import torch
import gordongames
import gym
import torch.nn.functional as F
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

class SequentialEnvironment:
    """
    The goal of a sequential environment is to seamlessly integrate
    Unity environments into workflows that already have gym environments.
    Unity environments can have multiple observations and can have
    multiple games within a single environment. gym environments need
    a wrapper to make their outputs look the same as the Unity env.
    """

    def __init__(self, env_type, preprocessor, seed=time.time(),
                                               **kwargs):
        """
        env_type: str
            the name of the environment
        preprocessor: function
            the preprocessing function to be used on each of the
            observations
        seed: int
            the random seed for the environment
        """
        self.env_type = env_type
        self.preprocessor = preprocessor
        self.seed = seed

        try:
            if "gordongames" in env_type or "nake" in env_type:
                self.env = gym.make(env_type, **kwargs)
            else:
                self.env = gym.make(env_type)
            self.env.seed(self.seed)
            self.is_gym = True
            self.raw_shape = self.env.reset().shape
        except:
            self.env = self.make_unity_env(env_type, seed=self.seed,
                                                **kwargs)
            self.is_gym = False
            self.raw_shape = self.env.reset()[0].shape
        self.action_space = self.env.action_space

    def make_unity_env(self, path, float_params=None, time_scale=1,
                                                      seed=time.time(),
                                                      **kwargs):
        """
        creates a gym environment from a unity game

        env_type: str
            the path to the game
        float_params: dict or None
            this should be a dict of argument settings for the unity
            environment
            keys: varies by environment
        time_scale: float
            argument to set Unity's time scale. This applies less to
            gym wrapped versions of Unity Environments, I believe..
            but I'm not sure
        seed: int
            the seed for randomness
        """
        path = os.path.expanduser(env_type)
        channel = EngineConfigurationChannel()
        env_channel = EnvironmentParametersChannel()
        env = UnityEnvironment(file_name=path,
                               side_channels=[channel,env_channel],
                               seed=seed)
        channel.set_configuration_parameters(time_scale = 1)
        env_channel.set_float_parameter("validation", 0)
        env_channel.set_float_parameter("egoCentered", 0)
        env = UnityToGymWrapper(env, allow_multiple_obs=True)
        return env

    def prep_obs(self, obs):
        """
        obs: list or ndarray
            the observation returned by the environment
        """
        if self.is_gym:
            prepped_obs = self.preprocessor(obs)
        else:
            prepped_obs = self.preprocessor(obs[0])
            # Handles the additional observations passed by the env
            if len(obs) > 1:
                prepped_obs = [prepped_obs, *obs[1:]]
        return prepped_obs

    def reset(self):
        obs = self.env.reset()
        return self.prep_obs(obs)

    def step(self,action):
        """
        action: list, vector, or int
            the action to take in this step. type can vary depending
            on the environment type
        """
        obs,rew,done,info = self.env.step(action)
        return self.prep_obs(obs), rew, done, info

    def get_action(self, preds):
        """
        Action data types can vary from evnironment to environment.
        This function handles converting outputs from the model
        to actions of the appropriate form for the environment.

        preds: torch tensor (..., N)
            the outputs from the model
        """
        if self.is_gym:
            probs = F.softmax(preds, dim=-1)
            action = sample_action(probs.data)
            return int(action.item())
        else:
            preds = preds.squeeze().cpu().data.numpy()
            return preds

    def render(self):
        """
        Calls the environment's render function if one exists
        """
        if hasattr(self.env, "render"): self.env.render()

class Runner:
    def __init__(self, datas, hyps, gate_q, stop_q, rew_q):
        """
        hyps - dict object with all necessary hyperparameters
                Keys (Assume string type keys):
                    "gamma" - reward decay coeficient
                    "n_tsteps" - number of steps to be taken in the
                                environment
                    "n_frame_stack" - number of frames to stack for
                                creation of the mdp state
                    "preprocessor" - function to preprocess raw observations
                    "env_type" - type of gym environment to be interacted 
                                with. Follows OpenAI's gym api.
        datas - dict of torch tensors with shared memory to collect data. Each 
                tensor contains indices from idx*n_tsteps to (idx+1)*n_tsteps
                Keys (assume string keys):
                    "states" - Collects the MDP states at each timestep t
                    "deltas" - Collects the gae deltas at timestep t+1
                    "rewards" - Collects float rewards collected at each timestep t
                    "dones" - Collects the dones collected at each timestep t
                    "actions" - Collects actions performed at each timestep t
                    If Using Recurrent Model:
                        "h_states" - Collects recurrent states at each timestep t
        gate_q - multiprocessing queue. Allows main process to control when
                rollouts should be collected.
        stop_q - multiprocessing queue. Used to indicate to main process that
                a rollout has been collected.
        rew_q - holds average reward over all processes
                type: multiprocessing Queue
        """

        self.hyps = hyps
        self.datas = datas
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.rew_q = rew_q
        self.obs_deque = deque(maxlen=hyps['n_frame_stack'])

    def run(self, net):
        """
        run is the entry function to begin collecting rollouts from the
        environment using the specified net. gate_q indicates when to begin
        collecting a rollout and is controlled from the main process.
        The stop_q is used to indicate to the main process that a new rollout
        has been collected.

        net - torch Module object. This is the model to interact with the
            environment.
        """
        self.net = net
        self.env = SequentialEnvironment(**self.hyps)
        state = next_state(self.env, self.obs_deque, obs=None, reset=True)
        self.state_bookmark = state
        self.h_bookmark = None
        if self.net.is_recurrent:
            self.h_bookmark = cuda_if(torch.zeros(1, self.net.h_size))
        self.ep_rew = 0
        #self.net.train(mode=False) # fixes potential batchnorm and dropout issues
        for p in self.net.parameters(): # Turn off gradient collection
            p.requires_grad = False
        while True:
            idx = self.gate_q.get() # Opened from main process
            self.rollout(self.net, idx, self.hyps)
            self.stop_q.put(idx) # Signals to main process that data has been collected

    def rollout(self, net, idx, hyps):
        """
        rollout handles the actual rollout of the environment for n steps in time.
        It is called from run and performs a single rollout, placing the
        collected data into the shared lists found in the datas dict.

        net - torch Module object. This is the model to interact with the
            environment.
        idx - int identification number distinguishing the
            portion of the shared array designated for this runner
        hyps - dict object with all necessary hyperparameters
                Keys (Assume string type keys):
                    "gamma" - reward decay coeficient
                    "n_tsteps" - number of steps to be taken in the
                                environment
                    "n_frame_stack" - number of frames to stack for
                                creation of the mdp state
        """
        state = self.state_bookmark
        h = self.h_bookmark
        n_tsteps = hyps['n_tsteps']
        startx = idx*n_tsteps
        prev_val = None
        for i in range(n_tsteps):
            self.datas['states'][startx+i] = cuda_if(torch.FloatTensor(state))
            state_in = self.datas['states'][startx+i].unsqueeze(0)
            if 'h_states' in self.datas:
                self.datas['h_states'][startx+i] = h.data[0]
                h_in = h.data
                val, logits, h = net(state_in, h_in)
            else:
                val, logits = net(state_in)
            probs = F.softmax(logits, dim=-1)
            action = sample_action(probs.data)
            action = int(action.item())
            obs, rew, done, info = self.env.step(action+hyps['action_shift'])
            if hyps['render']:
                self.env.render()
            self.ep_rew += rew
            reset = done
            if "Pong" in hyps['env_type'] and rew != 0:
                done = True
            if done:
                self.rew_q.put(.99*self.rew_q.get() + .01*self.ep_rew)
                self.ep_rew = 0
                # Reset Recurrence
                if h is not None:
                    h = cuda_if(torch.zeros(1,self.net.h_size))

            self.datas['rewards'][startx+i] = rew
            self.datas['dones'][startx+i] = float(done)
            self.datas['actions'][startx+i] = action
            state = next_state(self.env, self.obs_deque, obs=obs, reset=reset)
            if i > 0:
                prev_rew = self.datas['rewards'][startx+i-1]
                prev_done = self.datas['dones'][startx+i-1]
                delta = prev_rew + hyps['gamma']*val.data*(1-prev_done) - prev_val
                self.datas['deltas'][startx+i-1] = delta
            prev_val = val.data.squeeze()

        # Funky bootstrapping
        endx = startx+n_tsteps-1
        if not done:
            state_in = cuda_if(torch.FloatTensor(state)).unsqueeze(0)
            if 'h_states' in self.datas:
                val, logits, _ = net(state_in, h.data)
            else:
                val, logits = net(state_in)
            self.datas['rewards'][endx] += hyps['gamma']*val.squeeze() # Bootstrap
            self.datas['dones'][endx] = 1.
        self.datas['deltas'][endx] = self.datas['rewards'][endx] - prev_val
        self.state_bookmark = state
        if h is not None:
            self.h_bookmark = h.data

class StatsRunner:
    """
    This class is used to evaluate the performance of the model over
    multiple episodes of the environment. This allows us to collect
    a more accurate average reward estimate.
    """
    def __init__(self, hyps):
        """
        hyps - dict object with all necessary hyperparameters
            keys (Assume string type keys):
                "n_tsteps" - number of steps to be taken in the
                            environment
                "n_frame_stack" - number of frames to stack for
                            creation of the mdp state
                "preprocessor" - function to preprocess raw observations
                "env_type" - type of gym environment to be interacted 
                            with. Follows OpenAI's gym api.
                "seed" - the random seed for the env
        """
        self.hyps = hyps
        self.env = SequentialEnvironment(**self.hyps)
        self.obs_deque = deque(maxlen=self.hyps['n_frame_stack'])
        self.n_episodes = try_key(self.hyps, "n_test_eps", 15)

    def rollout(self, net):
        """
        this function performs the data collection

        net: torch Module
            the neural network to be used in the environment
        """
        print("Starting evaluation")
        state = next_state(self.env, self.obs_deque, obs=None, reset=True)
        h = None
        if net.is_recurrent:
            h = cuda_if(torch.zeros(1, net.h_size))
        ep_rew = 0
        net.eval() # fixes potential batchnorm and dropout issues
        prev_val = None
        ep_count = 0
        while ep_count < self.n_episodes:
            state_in = cuda_if(torch.FloatTensor(state))[None] #(1,C,H,W)
            if h is not None:
                h_in = h.data
                val, logits, h = net(state_in, h_in)
            else:
                val, logits = net(state_in)
            action = self.env.get_action(logits)

            a = action+self.hyps['action_shift']
            obs,rew,done,info = self.env.step(a)
            if self.hyps['render']:
                self.env.render()
            ep_rew += rew
            reset = done
            if "Pong" in self.hyps['env_type'] and rew != 0:
                done = True
            if done:
                ep_count += 1
                # Reset Recurrence
                if h is not None:
                    h = cuda_if(torch.zeros(1,net.h_size))

            state = next_state(self.env, self.obs_deque, obs=obs,
                                                         reset=reset)
        return ep_rew / ep_count



