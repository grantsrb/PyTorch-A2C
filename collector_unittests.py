import unittest
import numpy as np
from collector import Collector
import torch
from model import Model

class CollectorTests(unittest.TestCase):

    if torch.cuda.is_available():
        torch.FloatTensor = torch.cuda.FloatTensor
        torch.LongTensor = torch.cuda.LongTensor

    n_envs=1
    grid_size=[15,15]
    n_foods=1
    unit_size=10
    n_state_frames=3
    n_tsteps=15
    action_space = 4
    gamma = .99
    collector = Collector(n_envs=n_envs, grid_size=grid_size, n_foods=n_foods, unit_size=unit_size, n_state_frames=n_state_frames, n_tsteps=n_tsteps)
    prepped_shape = collector.prepped_shape
    state_shape = collector.state_shape
    net = Model(state_shape, action_space)


    def test_preprocess(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_state_frames=self.n_state_frames, net=self.net, n_tsteps=self.n_tsteps)
        observation = torch.FloatTensor(torch.zeros(*self.prepped_shape))
        prepped_obs = collector.preprocess(observation)
        self.assertTrue(np.array_equal(prepped_obs.size(), (self.prepped_shape[2], *self.prepped_shape[0:2])))

    def test_make_state_noprev(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_state_frames=self.n_state_frames, net=self.net, n_tsteps=self.n_tsteps)
        prepped_obs = torch.FloatTensor(torch.zeros(*self.prepped_shape))
        state = collector.make_state(prepped_obs)
        self.assertTrue(np.array_equal(state.size(), [1]+self.state_shape))

    def test_make_state(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_state_frames=self.n_state_frames, net=self.net, n_tsteps=self.n_tsteps)
        prepped_obs = torch.FloatTensor(torch.zeros(*self.prepped_shape))
        prev_state = torch.FloatTensor(torch.zeros(*self.state_shape))
        state = collector.make_state(prepped_obs, prev_state)
        self.assertTrue(np.array_equal(state.size(), [1]+self.state_shape))

    def test_softmax(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_state_frames=self.n_state_frames, net=self.net, n_tsteps=self.n_tsteps)
        zeros = np.zeros((1,self.action_space)).astype(np.float32)
        for i in range(len(zeros)):
            expected_output = zeros.copy()
            expected_output[0][i] = 1
            output = collector.softmax(expected_output)
            max_idx = np.argmax(output, -1)
            self.assertTrue(max_idx == i)

    def test_softmax_specific(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_state_frames=self.n_state_frames, net=self.net, n_tsteps=self.n_tsteps)
        for i in range(4):
            input_array = np.array([[np.random.randint(0,10) for i in range(4)]], dtype=np.float32)
            input_array[0][i] = 100
            output = collector.softmax(input_array)
            max_idx = np.argmax(output, -1)
            self.assertTrue(max_idx == i)

    def test_get_action(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_state_frames=self.n_state_frames, net=self.net, n_tsteps=self.n_tsteps)
        pi = torch.FloatTensor([[-10.0, 1000.0, -10.0, -10.0]])
        action = collector.get_action(pi)
        self.assertTrue(action == 1)

    def test_temporal_difference(self):
        collector = Collector(n_envs=self.n_envs, grid_size=self.grid_size, n_foods=self.n_foods, unit_size=self.unit_size, n_state_frames=self.n_state_frames, net=self.net, n_tsteps=self.n_tsteps, gamma = self.gamma)
        output = collector.temporal_difference(1, 1, 1)
        expected_output = 1 + self.gamma - 1
        self.assertTrue(expected_output == output)










if __name__ == "__main__":
    unittest.main()
