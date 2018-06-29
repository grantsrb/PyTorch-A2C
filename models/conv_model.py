import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Simple, sequential convolutional net.
'''

class ConvModel(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, input_space, output_space, h_size=288, bnorm=False):
        super(ConvModel, self).__init__()

        self.is_recurrent = False
        self.input_space = input_space
        self.output_space = output_space
        self.h_size = h_size
        self.bnorm = bnorm

        self.convs = nn.ModuleList([])

        # Embedding Net
        shape = input_space.copy()

        ksize=3; stride=1; padding=1; out_depth=16
        self.convs.append(self.conv_block(input_space[-3],out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=24
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=32
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=48
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=64
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                            stride=stride, padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize, padding=padding, stride=stride)
        
        self.features = nn.Sequential(*self.convs)
        self.flat_size = int(np.prod(shape))
        print("Flat Features Size:", self.flat_size)
        self.resize_emb = nn.Sequential(nn.Linear(self.flat_size, self.h_size), nn.ReLU())

        # Policy
        self.emb_bnorm = nn.BatchNorm1d(self.h_size)
        self.pi = nn.Linear(self.h_size, self.output_space)
        self.value = nn.Linear(self.h_size, 1)

    def get_new_shape(self, shape, depth, ksize, padding, stride):
        new_shape = [depth]
        for i in range(2):
            new_shape.append(self.new_size(shape[i+1], ksize, padding, stride))
        return new_shape
        
    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def forward(self, x):
        embs = self.emb_net(x)
        val, pi = self.policy(embs)
        return val, pi

    def emb_net(self, state):
        """
        Creates an embedding for the state.

        state - Variable FloatTensor with shape (BatchSize, Channels, Height, Width)
        """
        feats = self.features(state)
        feats = feats.view(feats.shape[0], -1)
        state_embs = self.resize_emb(feats)
        return state_embs

    def policy(self, state_emb):
        """
        Uses the state embedding to produce an action.

        state_emb - the state embedding created by the emb_net
        """
        if self.bnorm:
            state_emb = self.emb_bnorm(state_emb)
        pi = self.pi(state_emb)
        value = self.value(state_emb)
        return value, pi

    def conv_block(self, chan_in, chan_out, ksize=3, stride=1, padding=1, activation="lerelu", max_pool=False, bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride=stride, padding=padding))
        if activation is not None: activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "lerelu" in activation:
            block.append(nn.LeakyReLU(negative_slope=.05))
        elif "selu" in activation:
            block.append(nn.SELU())
        if max_pool:
            block.append(nn.MaxPool2d(2, 2))
        if bnorm:
            block.append(nn.BatchNorm2d(chan_out))
        return nn.Sequential(*block)

    def dense_block(self, chan_in, chan_out, activation="relu", bnorm=True):
        block = []
        block.append(nn.Linear(chan_in, chan_out))
        if activation is not None: activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "lerelu" in activation:
            block.append(nn.LeakyReLU())
        elif "selu" in activation:
            block.append(nn.SELU())
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
        noise = self.cuda_if(torch.normal(means,std=std))
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
        noise = self.cuda_if(torch.normal(means,std=std))
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

