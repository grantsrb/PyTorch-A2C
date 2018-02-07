import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Simple, sequential convolutional net.
'''

class Model(nn.Module):
    def __init__(self, input_space, output_space):
        super(Model, self).__init__()

        self.input_space = input_space
        self.output_space = output_space

        self.convs = nn.ModuleList([])
        self.dropouts = nn.ModuleList([]) # Used in dense block

        self.conv1 = self.conv_block(input_space[-3],8)
        self.convs.append(self.conv1)
        self.conv2 = self.conv_block(8, 8, stride=2, bnorm=True)
        self.convs.append(self.conv2)
        self.conv3 = self.conv_block(8, 12, bnorm=True)
        self.convs.append(self.conv3)

        self.features = nn.Sequential(*self.convs)
        self.classifier = None

        self.flat_shape = None


    def conv_block(self, chan_in, chan_out, ksize=3, stride=1, padding=1, activation="relu", max_pool=False, bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride, padding=padding))
        activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "selu" in activation:
            block.append(nn.SELU())
        if max_pool:
            block.append(nn.MaxPool2d(2, 2))
        if bnorm:
            block.append(nn.BatchNorm2d(chan_out))
        return nn.Sequential(*block)

    def dense_block(self, chan_in, chan_out, dropout_p=0, activation="relu", batch_norm=True):
        block = []
        dropout = nn.Dropout(dropout_p)
        block.append(dropout)
        self.dropouts.append(dropout)
        block.append(nn.Linear(chan_in, chan_out))
        activation=activation.lower()
        if "relu" in activation:
            block.append(nn.ReLU())
        elif "tanh" in activation:
            block.append(nn.Tanh())
        elif "elu" in activation:
            block.append(nn.ELU())
        elif "selu" in activation:
            block.append(nn.SELU())
        if batch_norm:
            block.append(nn.BatchNorm1d(chan_out))
        return nn.Sequential(*block)

    def forward(self, x):
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        if self.classifier is None:
            self.flat_shape = feats.shape
            modules = [self.dense_block(feats.size(1), 200, batch_norm=True)]
            modules.append(self.dense_block(200, 200, batch_norm=True))
            self.precursor = nn.Sequential(*modules)
            self.classifier = self.dense_block(200,self.output_space,activation="none",batch_norm=False)
            self.evaluator = self.dense_block(200, 1, activation="none", batch_norm=False)
        feats = self.precursor(feats)
        return self.evaluator(feats), self.classifier(feats)

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

    @staticmethod
    def preprocess(pic, env_type='snake-v0'):
        if env_type == "Pong-v0":
            pic = pic[35:195] # crop
            pic = pic[::2,::2,0] # downsample by factor of 2
            pic[pic == 144] = 0 # erase background (background type 1)
            pic[pic == 109] = 0 # erase background (background type 2)
            pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
        if env_type == "snake-v0":
            new_pic = np.zeros(pic.shape,dtype=np.float32)
            new_pic[:,:,0][pic[:,:,0]==1] = 1
            new_pic[:,:,0][pic[:,:,0]==255] = 1.5
            new_pic[:,:,1][pic[:,:,1]==255] = 0
            new_pic[:,:,2][pic[:,:,2]==255] = .3
            pic = np.sum(new_pic, axis=-1)
        return pic[None]
