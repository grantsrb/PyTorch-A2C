import unittest
import numpy as np
from collector import Collector
import torch
import dense_model as model

class CollectorTests(unittest.TestCase):

    if torch.cuda.is_available():
        torch.FloatTensor = torch.cuda.FloatTensor
        torch.LongTensor = torch.cuda.LongTensor

    env_type='snake-v0'
    n_envs=1
    grid_size=[15,15]
    n_foods=1
    unit_size=10
    n_obs_stack=2
    n_tsteps=15
    action_space = 4
    gamma = .99
    collector = Collector(n_envs=n_envs, grid_size=grid_size, n_foods=n_foods, unit_size=unit_size, n_obs_stack=n_obs_stack, net=None, n_tsteps=n_tsteps, env_type=env_type, preprocessor=model.Model.preprocess)
    obs_shape = collector.obs_shape
    prepped_shape = collector.prepped_shape
    state_shape = collector.state_shape
    net = model.Model(state_shape, action_space, env_type=env_type)


    def test_preprocess(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_obs_stack=self.n_obs_stack, net=self.net, n_tsteps=self.n_tsteps, env_type=self.env_type, preprocessor=model.Model.preprocess)
        dummy_obs = collector.envs[0].reset()
        obs_shape = dummy_obs.shape
        self.assertTrue(np.array_equal(obs_shape, self.obs_shape))
        prepped_obs = collector.preprocess(dummy_obs)
        self.assertTrue(np.array_equal(prepped_obs.shape, self.prepped_shape))
        for i in range(prepped_obs.shape[0]):
            for j in range(prepped_obs.shape[1]):
                self.assertTrue(prepped_obs[i,j] <= 1.5 and prepped_obs[i,j] >= 0)

    def test_make_state_noprev(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_obs_stack=self.n_obs_stack, net=self.net, n_tsteps=self.n_tsteps, env_type=self.env_type, preprocessor=model.Model.preprocess)
        prepped_obs = np.ones(self.prepped_shape)
        state = collector.make_state(prepped_obs)
        self.assertTrue(np.array_equal(state.shape, [2,self.prepped_shape[-1]]))
        for i in range(state.shape[1]):
            self.assertTrue(state[0,i] == 1)
            self.assertTrue(state[1,i] == 0)

    def test_make_state(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_obs_stack=self.n_obs_stack, net=self.net, n_tsteps=self.n_tsteps, env_type=self.env_type, preprocessor=model.Model.preprocess)
        prepped_obs = np.ones(self.prepped_shape)
        prev_state = np.ones(self.state_shape)*2
        state = collector.make_state(prepped_obs, prev_state)
        self.assertTrue(np.array_equal(state.shape, [2,self.prepped_shape[-1]]))
        for i in range(state.shape[1]):
            self.assertTrue(state[0,i] == 1)
            self.assertTrue(state[1,i] == 2)

    def test_softmax(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_obs_stack=self.n_obs_stack, net=self.net, n_tsteps=self.n_tsteps, env_type=self.env_type, preprocessor=model.Model.preprocess)
        zeros = np.zeros((1,self.action_space)).astype(np.float32)
        for i in range(len(zeros)):
            expected_output = zeros.copy()
            expected_output[0][i] = 1
            output = collector.softmax(expected_output)
            max_idx = np.argmax(output, -1)
            self.assertTrue(max_idx == i)

    def test_softmax_specific(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_obs_stack=self.n_obs_stack, net=self.net, n_tsteps=self.n_tsteps, env_type=self.env_type, preprocessor=model.Model.preprocess)
        for i in range(4):
            input_array = np.array([[np.random.randint(0,10) for i in range(4)]], dtype=np.float32)
            input_array[0][i] = 100
            output = collector.softmax(input_array)
            max_idx = np.argmax(output, -1)
            self.assertTrue(max_idx == i)

    def test_get_action(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_obs_stack=self.n_obs_stack, net=self.net, n_tsteps=self.n_tsteps, env_type=self.env_type, preprocessor=model.Model.preprocess)
        pi = torch.FloatTensor([[-10.0, 1000.0, -10.0, -10.0]])
        action = collector.get_action(pi)
        self.assertTrue(action == 1)











if __name__ == "__main__":
    unittest.main()
