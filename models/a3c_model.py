import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class A3CModel(nn.Module):
    def __init__(self, input_space, output_space, emb_size=256, bnorm=False):
        super(Model, self).__init__()

        self.input_space = input_space
        self.output_space = output_space
        self.emb_size = emb_size

        # Embedding Net
        self.convs = nn.ModuleList([])
        shape = input_space.copy()
        self.conv1 = self.conv_block(input_space[-3],16, ksize=8, stride=4, padding=0, bnorm=bnorm, activation='relu')
        self.convs.append(self.conv1)
        shape = self.new_shape(shape, 16, ksize=8, stride=4, padding=0)

        self.conv2 = self.conv_block(16, 32, ksize=4, stride=2, padding=0, bnorm=bnorm, activation='relu')
        self.convs.append(self.conv2)
        shape = self.new_shape(shape, 32, ksize=4, stride=2, padding=0)

        self.features = nn.Sequential(*self.convs)
        self.flat_size = int(np.prod(shape))
        self.proj_matrx = nn.Linear(self.flat_size, self.emb_size)

        # Policy
        self.emb_bnorm = nn.BatchNorm1d(self.emb_size)
        self.pi = nn.Linear(self.emb_size, self.output_space)
        self.value = nn.Linear(self.emb_size, 1)

    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def new_shape(self, shape, depth, ksize=3, padding=1, stride=2):
        shape[-1] = self.new_size(shape[-1], ksize=ksize, padding=padding, stride=stride)
        shape[-2] = self.new_size(shape[-2], ksize=ksize, padding=padding, stride=stride)
        shape[-3] = depth
        return shape

    def forward(self, x, bnorm=False):
        embs = self.encoder(x)
        val, pi = self.policy(embs, bnorm=bnorm)
        return val, pi

    def encoder(self, state):
        """
        Creates an embedding for the state.

        state - Variable FloatTensor with shape (BatchSize, Channels, Height, Width)
        """
        feats = self.features(state)
        feats = feats.view(feats.shape[0], -1)
        feats = self.proj_matrx(feats)
        return feats

    def policy(self, state_emb, bnorm=True):
        """
        Uses the state embedding to produce an action.

        state_emb - the state embedding created by the encoder
        """
        if bnorm:
            state_emb = self.emb_bnorm(state_emb)
        pi = self.pi(state_emb)
        value = self.value(Variable(state_emb.data))
        return value, pi

    def conv_block(self, chan_in, chan_out, ksize=3, stride=1, padding=1, activation="relu", max_pool=False, bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride=stride, padding=padding))
        if activation is not None: 
            activation=activation.lower()
        if "relu" == activation:
            block.append(nn.ReLU())
        elif "selu" == activation:
            block.append(nn.SELU())
        elif "elu" == activation:
            block.append(nn.ELU())
        elif "tanh" == activation:
            block.append(nn.Tanh())
        if max_pool:
            block.append(nn.MaxPool2d(2, 2))
        if bnorm:
            block.append(nn.BatchNorm2d(chan_out))
        return nn.Sequential(*block)

    def dense_block(self, chan_in, chan_out, activation="relu", bnorm=True):
        block = []
        block.append(nn.Linear(chan_in, chan_out))
        if activation is not None: activation=activation.lower()
        if "relu" == activation:
            block.append(nn.ReLU())
        elif "selu" == activation:
            block.append(nn.SELU())
        elif "elu" == activation:
            block.append(nn.ELU())
        elif "tanh" == activation:
            block.append(nn.Tanh())
        if bnorm:
            block.append(nn.BatchNorm1d(chan_out))
        return nn.Sequential(*block)

    def add_noise(self, x, mean=0.0, std=0.01):
        """
        Adds a normal distribution over the entries in a matrix.
        """
        means = torch.zeros(*x.size()).float()
        if mean != 0.0:
            means = means + mean
        noise = torch.normal(means,std=std)
        if type(x) == type(Variable()):
            noise = Variable(noise)
        return x+noise

    def multiply_noise(self, x, mean=1, std=0.01):
        """
        Multiplies a normal distribution over the entries in a matrix.
        """
        means = torch.zeros(*x.size()).float()
        if mean != 0:
            means = means + mean
        noise = torch.normal(means,std=std)
        if type(x) == type(Variable()):
            noise = Variable(noise)
        return x*noise

    def req_grads(self, calc_bool):
        """
        An on-off switch for the requires_grad parameter for each internal Parameter.

        calc_bool - Boolean denoting whether gradients should be calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking into pytorch...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print("NaNs in Grad!")

