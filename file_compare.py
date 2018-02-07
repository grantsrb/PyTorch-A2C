import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Updater():
    """
    This class converts the data collected from the rollouts into useable data to update
    the model. The main function to use is calc_loss which accepts a rollout to
    add to the global loss of the model. The model isn't updated, however, until calling
    calc_gradients followed by update_model. If the size of the epoch is restricted by the memory, you can call calc_gradients to clear the graph.
    """

    def __init__(self, net, lr, entropy_const=0.01, value_const=0.5, gamma=0.99, _lambda=0.98, max_norm=0.5):
        self.net = net
        self.optim = optim.RMSprop(self.net.parameters(), lr=lr)
        self.global_loss = 0 # Used for efficiency in backprop
        self.entropy_const = entropy_const
        self.value_const = value_const
        self.gamma = gamma
        self._lambda = _lambda
        self.max_norm = max_norm

        # Tracking variables
        self.pg_loss = 0
        self.value_loss = 0
        self.entropy = 0
        self.loss_count = 0
        self.info = {}

    def calc_gradients(self):
        """
        Calculates the gradients of each parameter within the net from the global_loss
        Variable.

        Returns the loss as a float
        """

        try:
            self.global_loss.backward()
            self.norm = nn.utils.clip_grad_norm(self.net.parameters(), self.max_norm)
            if self.loss_count > 0:
                self.info = {"Global Loss":self.global_loss.data[0]/self.loss_count,
                            "Policy Loss":self.pg_loss/self.loss_count,
                            "Value Loss":self.value_loss/self.loss_count,
                            "Entropy":self.entropy/self.loss_count,
                            "Norm":self.norm}
            self.global_loss = 0
            self.pg_loss = 0
            self.value_loss = 0
            self.entropy = 0
            self.loss_count = 0
        except RuntimeError:
            print("Attempted to use self.global_loss.backward() when no graph is created yet!")

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

        states = Variable(torch.FloatTensor(states))
        values, raw_pis = self.net.forward(states)
        softlog_pis = F.log_softmax(raw_pis, dim=-1)
        softlog_column = softlog_pis[list(range(softlog_pis.size(0))), actions]
        advantages = self.discount(advantages, dones, self.gamma*self._lambda)
        advantages = self.normalize(advantages)
        pg_step = softlog_column*Variable(torch.FloatTensor(advantages))
        pg_loss = -torch.mean(pg_step)

        value_targets = self.discount(rewards, dones, self.gamma)
        value_loss =self.value_const*F.mse_loss(values.squeeze(),Variable(torch.FloatTensor(value_targets)))

        softmaxes = F.softmax(raw_pis, dim=-1)
        entropy_step = softmaxes*softlog_pis
        entropy = -self.entropy_const*torch.mean(entropy_step)

        self.global_loss += pg_loss + value_loss - entropy
        self.pg_loss += pg_loss.data[0]
        self.value_loss += value_loss.data[0]
        self.entropy += entropy.data[0]
        self.loss_count += 1

    def discount(self, array, mask, discount_factor):
        """
        Dicounts the argued array following the bellman equation.

        array - array to be discounted
        mask - binary array denoting the end of an episode
        discount_factor - float between 0 and 1 used to discount the reward

        Returns the discounted array as an ndarray of type np.float32
        """

        running_sum = 0
        discounts = np.zeros(len(array))
        for i in reversed(range(len(array))):
            if mask[i] == 1: running_sum = 0
            running_sum = array[i] + discount_factor*running_sum
            discounts[i] = running_sum
        return discounts

    def normalize(self, array, mean=None, std=None):
        """
        Normalizes the array's values. Optionally pass a specific mean or standard
        deviation to use for normalization.

        array - 1 dimensional ndarray or torch FloatTensor
        mean - optional float denoting the mean to use for normalization.
                If None, the mean will be calculated from the array
        std - optional float denoting the standard deviation to use for
              normalization. If None, the standard deviation will be
              calculated from the array.
        """

        if mean is None:
            if type(array) == type(np.array([])):
                mean = np.mean(array)
            else:
                mean = torch.mean(array)
        if std is None:
            if type(array) == type(np.array([])):
                std = np.std(array)
            else:
                std = torch.std(array)

        return (array - mean)/(std+1e-7)

    def print_statistics(self):
        print(" – ".join([key+": "+str(round(val,5)) if "ntropy" not in key else key+": "+str(val) for key,val in self.info.items()]))

    def save_model(self, net_file_name, optim_file_name):
        """
        Saves the state dict of the model to file.

        file_name - string name of the file to save the state_dict to
        """
        torch.save(self.net.state_dict(), net_file_name)
        torch.save(self.optim.state_dict(), optim_file_name)

    def update_model(self):
        """
        Performs backprop on any residual graphs and then updates the model using gradient
        descent.
        """

        self.calc_gradients()

        self.optim.step()
        self.optim.zero_grad()
