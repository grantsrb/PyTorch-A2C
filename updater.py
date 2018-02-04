import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np

class Updater():
    """
    This class converts the data collected from the rollouts into useable data to update
    the model. The main function to use is calc_loss which accepts a rollout to
    add to the global loss of the model. The model isn't updated, however, until calling
    calc_gradients followed by update_model. If the size of the epoch is restricted by the memory, you can call calc_gradients to clear the graph.
    """

    def __init__(self, net, lr, entropy_const=0.01, value_const=0.5, gamma=0.99, _lambda=0.98):
        self.net = net
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.global_loss = 0 # Used for efficiency in backprop
        self.entropy_const = entropy_const
        self.value_const = value_const
        self.gamma = gamma
        self._lambda = _lambda

    def calc_gradients(self):
        """
        Calculates the gradients of each parameter within the net from the global_loss
        Variable.
        """

        try:
            loss = self.global_loss[0]
            self.global_loss.backward()
            self.global_loss = 0
            return loss
        except RuntimeError:
            return 0

    def calc_loss(self, states, rewards, dones, actions, advantages):
        """
        This function accepts the data collected from a rollout, calculates the loss
        associated with the rollout, and then adds it to the global_loss.

        states - torch FloatTensor of the environment states from the rollouts
                shape = (n_states, *state_shape)
        rewards - torch FloatTensor of rewards from the rollouts
                shape = (n_states,)
        dones - torch FloatTensor of done signals from the rollouts
                dones = (n_states,)
        actions - torch LongTensor or python integer list denoting the actual action
                indexes taken in the rollout
        """

        values, raw_pis = self.net.forward(states)
        softlog_pis = F.log_softmax(raw_pis, dim=-1)
        softlog_pis = softlog_pis[list(range(softlog_pis.size(0))), actions]
        advantages = self.discount(advantages, dones, self.gamma*self._lambda)
        action_loss = softlog_pis*advantages

        value_targets = self.discount(rewards, dones, self.gamma)
        value_loss = self.value_const*F.mse_loss(values, Variable(discounted_rewards))

        entropy = self.entropy_const*-torch.mean(F.softmax(raw_pis, dim=-1)*softlog_pis)

        self.global_loss += action_loss + value_loss + entropy

    def discount(self, array, mask, discount_factor):
        """
        Dicounts the argued array following the bellman equation.

        array - array to be discounted
        mask - binary array denoting the end of an episode
        discount_factor - float between 0 and 1 used to discount the reward
        """

        running_sum = 0
        discounts = [0]*len(array)
        for i in reversed(range(len(array))):
            if mask[i] == 1: running_sum = 0
            running_sum = array[i] + discount_factor*running_sum
            discounts[i] = running_sum
        return discounts

    def update_model(self):
        """
        Calculates any residual gradients and then updates the model.
        """

        loss = self.calc_gradients()

        self.optim.step()
        self.optim.zero_grad()
