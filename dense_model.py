import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

"""
Description: A simple feedforward model with batchnorm.
"""

class Model(nn.Module):
    def __init__(self, input_space, output_space):
        super(Model, self).__init__()
        self.hidden_dim = 200
        self.input_space = input_space
        self.output_space = output_space

        self.entry = nn.Linear(input_space[-1], self.hidden_dim)
        self.bnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bnorm2 = nn.BatchNorm1d(self.hidden_dim)

        self.action_out = nn.Linear(self.hidden_dim, output_space)
        self.value_out = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, bnorm=True):
        x.data[:,1,:] = -x.data[:,1,:]
        x = torch.sum(x.data, dim=1)
        fx = F.relu(self.entry(Variable(x)))
        if bnorm: fx = self.bnorm1(fx)
        fx = F.relu(self.hidden(fx))
        if bnorm: fx = self.bnorm2(fx)
        action = self.action_out(fx)
        value = self.value_out(fx)
        return value, action

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking into pytorch...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print(param)

    def req_grads(self, grad_on):
        """
        Used to turn off and on all gradient calculation requirements for speed.

        grad_on - bool denoting whether gradients should be calculated
        """
        for p in self.parameters():
            p.requires_grad = grad_on

    @staticmethod
    def preprocess(pic, env_type='snake-v0'):
        if env_type == "Pong-v0":
            pic = pic[35:195] # crop
            pic = pic[::2,::2,0] # downsample by factor of 2
            pic[pic == 144] = 0 # erase background (background type 1)
            pic[pic == 109] = 0 # erase background (background type 2)
            pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
        elif env_type == "snake-v0":
            new_pic = np.zeros(pic.shape,dtype=np.float32)
            new_pic[:,:,0][pic[:,:,0]==1] = 1
            new_pic[:,:,0][pic[:,:,0]==255] = 1.5
            new_pic[:,:,1][pic[:,:,1]==255] = 0
            new_pic[:,:,2][pic[:,:,2]==255] = .3
            pic = np.sum(new_pic, axis=-1)
        return pic.ravel()[None]
