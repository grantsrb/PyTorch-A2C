import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

"""
Description: A simple feedforward model with batchnorm.
"""

class FCModel(nn.Module):
    def __init__(self, input_shape, output_space, h_size=200, bnorm=False):
        super(FCModel, self).__init__()

        self.flat_size = np.prod(input_shape[-3:])

        block = [nn.Linear(self.flat_size, h_size)]
        if bnorm:
            block.append(nn.BatchNorm1d(h_size))
        block.append(nn.ReLU())
        block.append(nn.Linear(h_size, h_size))
        if bnorm:
            block.append(nn.BatchNorm1d(h_size))

        self.base = nn.Sequential(*block)

        self.action_out = nn.Linear(h_size, output_space)
        self.value_out = nn.Linear(h_size, 1)

    def forward(self, x):
        fx = x.view(len(x), -1)
        fx = self.base(fx)
        action = self.action_out(fx)
        value = self.value_out(fx)
        return value, action

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking into pytorch...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print("NaNs in Grad!")

    def req_grads(self, grad_on):
        """
        Used to turn off and on all gradient calculation requirements for speed.

        grad_on - bool denoting whether gradients should be calculated
        """
        for p in self.parameters():
            p.requires_grad = grad_on
