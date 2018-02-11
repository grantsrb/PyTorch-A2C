from updater import Updater
import unittest
import torch
import numpy as np
from dense_model import Model
from torch.autograd import Variable

class UpdaterTests(unittest.TestCase):

    if torch.cuda.is_available():
        torch.FloatTensor = torch.cuda.FloatTensor
        torch.LongTensor = torch.cuda.LongTensor
    torch.manual_seed(1)

    n_envs=1
    grid_size=[15,15]
    n_foods=1
    unit_size=10
    n_obs_stack=2
    n_tsteps=15
    action_space = 4
    gamma = .99
    _lambda = .99
    lr = 0.001
    value_const = 0.5
    entropy_const = 0.01
    state_shape = (2,600)
    net = Model(state_shape, action_space, env_type='snake-v0')
    test_data = torch.FloatTensor(torch.ones(2,*state_shape))
    _,__ = net.forward(Variable(test_data))


    def test_calc_gradients(self):
        updater = Updater(self.net, self.lr, entropy_const=self.entropy_const, value_const=self.value_const, gamma=self.gamma, _lambda=self._lambda)

        net = Model(self.state_shape, self.action_space, env_type='snake-v0')
        _,__ = net.forward(Variable(self.test_data))
        net.load_state_dict(updater.net.state_dict())

        for p1,p2 in zip(updater.net.named_parameters(), net.named_parameters()):
            self.assertTrue(np.array_equal(p1[1].data.numpy(), p2[1].data.numpy()))

        v,p = updater.net.forward(Variable(self.test_data.clone()))
        updater.global_loss = torch.sum(v) + torch.sum(p)
        updater.calc_gradients()

        v,p = net.forward(Variable(self.test_data.clone()))
        loss = torch.sum(v) + torch.sum(p)
        loss.backward()

        for p1,p2 in zip(updater.net.named_parameters(), net.named_parameters()):
            self.assertTrue(np.array_equal(p1[1].grad.data.numpy(), p2[1].grad.data.numpy()))

    def test_discount(self):
        updater = Updater(self.net, self.lr, entropy_const=self.entropy_const, value_const=self.value_const, gamma=self.gamma, _lambda=self._lambda)
        array = [1,1,1]
        mask = [0,0,0]
        expected_output = [1+1.99*self.gamma, 1+1*self.gamma, 1]
        output = updater.discount(array, mask, self.gamma)
        self.assertTrue(np.array_equal(expected_output, output))

    def test_discount_mask(self):
        updater = Updater(self.net, self.lr, entropy_const=self.entropy_const, value_const=self.value_const, gamma=self.gamma, _lambda=self._lambda)
        array = [1,1,1]
        mask = [0,1,0]
        expected_output = [1.99, 1, 1]
        output = updater.discount(array, mask, self.gamma)
        self.assertTrue(np.array_equal(expected_output, output))


if __name__ == "__main__":
    unittest.main()
