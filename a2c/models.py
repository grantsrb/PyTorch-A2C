import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class A3CModel(nn.Module):
    def __init__(self, input_space, output_space, h_size=256,
                                                  bnorm=False,
                                                  is_discrete=True,
                                                  **kwargs):
        super().__init__()

        self.is_recurrent = False
        self.input_space = input_space
        self.output_space = output_space
        self.h_size = h_size
        self.is_discrete = is_discrete

        # Embedding Net
        self.convs = nn.ModuleList([])
        shape = input_space.copy()
        self.conv1 = self.conv_block(input_space[-3],16,ksize=8,
                                                    stride=4,
                                                    padding=0,
                                                    bnorm=bnorm,
                                                    activation='relu')
        self.convs.append(self.conv1)
        shape = self.new_shape(shape, 16, ksize=8, stride=4, padding=0)

        self.conv2 = self.conv_block(16, 32, ksize=4, stride=2,
                                                      padding=0,
                                                      bnorm=bnorm,
                                                      activation='relu')
        self.convs.append(self.conv2)
        shape = self.new_shape(shape, 32, ksize=4, stride=2, padding=0)

        self.features = nn.Sequential(*self.convs)
        self.flat_size = int(np.prod(shape))
        self.proj_matrx = nn.Linear(self.flat_size, self.h_size)

        # Policy
        self.emb_bnorm = nn.BatchNorm1d(self.h_size)
        outsize = self.output_space if self.is_discrete\
                                    else 2*self.output_space
        self.pi = nn.Linear(self.h_size, outsize)
        self.value = nn.Linear(self.h_size, 1)

    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def new_shape(self, shape, depth, ksize=3, padding=1, stride=2):
        shape[-1]=self.new_size(shape[-1],ksize=ksize,padding=padding,
                                                      stride=stride)
        shape[-2]=self.new_size(shape[-2], ksize=ksize,padding=padding,
                                                      stride=stride)
        shape[-3] = depth
        return shape

    def forward(self, x, bnorm=False):
        embs = self.encoder(x)
        val, pi = self.policy(embs, bnorm=bnorm)
        return val, pi

    def encoder(self, state):
        """
        Creates an embedding for the state.

        state - Variable FloatTensor (B, C, H, W)
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
        if not self.is_discrete:
            mu,sigma = torch.chunk(pi,2,dim=-1)
            sigma = F.softplus(sigma)+0.0001
            return value, (mu,sigma)
        return value, pi

    def conv_block(self, chan_in, chan_out, ksize=3, stride=1,
                                                     padding=1,
                                                     activation="relu",
                                                     max_pool=False,
                                                     bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride=stride,
                                                     padding=padding))
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

    def dense_block(self, chan_in, chan_out, activation="relu",
                                             bnorm=True):
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
        An on-off switch for the requires_grad parameter for each
        internal Parameter.

        calc_bool - Boolean denoting whether gradients should be
                    calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of
        sneaking in...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print("NaNs in Grad!")

class ConvModel(nn.Module):
    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, input_space, output_space, h_size=288,
                                                  bnorm=False,
                                                  is_discrete=True,
                                                  **kwargs):
        super().__init__()

        self.is_recurrent = False
        self.input_space = input_space
        self.output_space = output_space
        self.h_size = h_size
        self.bnorm = bnorm
        self.is_discrete = is_discrete

        self.convs = nn.ModuleList([])

        # Embedding Net
        shape = input_space.copy()

        ksize=3; stride=1; padding=1; out_depth=16
        self.convs.append(self.conv_block(input_space[-3],out_depth,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                           padding=padding,
                                           stride=stride)

        ksize=3; stride=1; padding=1; in_depth=out_depth
        out_depth=24
        self.convs.append(self.conv_block(in_depth,out_depth,ksize=ksize,
                                          stride=stride,padding=padding,
                                          bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                                     padding=padding,
                                                     stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=32
        self.convs.append(self.conv_block(in_depth,out_depth,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                                     padding=padding,
                                                     stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=64
        self.convs.append(self.conv_block(in_depth,out_depth,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                                     padding=padding,
                                                     stride=stride)

        self.features = nn.Sequential(*self.convs)
        self.flat_size = int(np.prod(shape))
        print("Flat Features Size:", self.flat_size)
        conv_h = 2000
        self.resize_emb = nn.Sequential(
                                nn.Linear(self.flat_size, conv_h),
                                nn.ReLU())
        # Policy
        block = []
        if self.bnorm: block.append(nn.BatchNorm1d(self.h_size))
        block.append(nn.Linear(conv_h, self.h_size))
        block.append(nn.ReLU())
        outsize = self.output_space if self.is_discrete\
                                else 2*self.output_space
        block.append(nn.Linear(self.h_size,outsize))
        self.pi = nn.Sequential(*block)
        # Value
        block = []
        if self.bnorm: block.append(nn.BatchNorm1d(self.h_size))
        block.append(nn.Linear(conv_h, self.h_size))
        block.append(nn.ReLU())
        block.append(nn.Linear(self.h_size,1))
        self.value = nn.Sequential(*block)

    def get_new_shape(self, shape, depth, ksize, padding, stride):
        new_shape = [depth]
        for i in range(2):
            new_shape.append(self.new_size(shape[i+1], ksize, padding,
                                                              stride))
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
        pi = self.pi(state_emb)
        value = self.value(state_emb)
        if not self.is_discrete:
            mu,sigma = torch.chunk(pi,2,dim=-1)
            sigma = F.softplus(sigma)+0.0001
            return value, (mu,sigma)
        return value, pi

    def conv_block(self, chan_in, chan_out, ksize=3, stride=1,
                                                     padding=1,
                                                     activation="ReLU",
                                                     max_pool=False,
                                                     bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride=stride,
                                                      padding=padding))
        if activation is not None:
            block.append(getattr(nn, activation)())
        if max_pool:
            block.append(nn.MaxPool2d(2, 2))
        if bnorm:
            block.append(nn.BatchNorm2d(chan_out))
        return nn.Sequential(*block)

    def dense_block(self, chan_in, chan_out, activation="ReLU",
                                             bnorm=True):
        block = []
        block.append(nn.Linear(chan_in, chan_out))
        if activation is not None:
            block.append(getattr(nn, activation)())
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
        An on-off switch for the requires_grad parameter for each
                                              internal Parameter.

        calc_bool - Boolean denoting whether gradients should be
                                                      calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

class FCModel(nn.Module):
    def __init__(self, input_shape, output_space, h_size=200,
                                                bnorm=False,
                                                is_discrete=True,
                                                **kwargs):
        super(FCModel, self).__init__()

        self.is_discrete = is_discrete
        self.is_recurrent = False
        self.flat_size = np.prod(input_shape[-3:])
        self.input_shape = input_shape
        self.output_space = output_space

        block = [nn.Linear(self.flat_size, h_size)]
        if bnorm:
            block.append(nn.BatchNorm1d(h_size))
        block.append(nn.ReLU())
        block.append(nn.Linear(h_size, h_size))
        if bnorm:
            block.append(nn.BatchNorm1d(h_size))

        self.base = nn.Sequential(*block)

        outsize = output_space if is_discrete else 2*output_space
        self.action_out = nn.Linear(h_size, outsize)
        self.value_out = nn.Sequential(nn.LayerNorm(h_size),
                                       nn.Linear(h_size, 1),
                                       nn.Linear(1,1))

    def forward(self, x):
        fx = x.view(len(x), -1)
        fx = self.base(fx)
        pi = self.action_out(fx)
        value = self.value_out(fx)
        if not self.is_discrete:
            mu,sigma = torch.chunk(pi,2,dim=-1)
            sigma = F.softplus(sigma)+0.0001
            return value, (mu,sigma)
        return value, pi

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of
        sneaking into pytorch...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print("NaNs in Grad!")

    def req_grads(self, grad_on):
        """
        Used to turn off and on all gradient calculation requirements
        for speed.

        grad_on - bool denoting whether gradients should be calculated
        """
        for p in self.parameters():
            p.requires_grad = grad_on

class GRU(nn.Module):
    """
    GRU units follow the formulae:

    z = sigmoid(W_x_z.mm(x) + W_h_z.mm(old_h) + b_z)
    r = sigmoid(W_x_r.mm(x) + W_h_r.mm(old_h) + b_r
    h = z*old_h + (1-z)*tanh(W_x_h.mm(x) + W_h_h.mm(r*old_h) + b_h)

    Where x is the new, incoming data and old_h is the h at the
    previous time step. Each of the W_x_ and W_h_ terms are weight
    matrices and the b_ terms are biases. In this implementation, all
    of the W_x_ terms are combined into a single variable. Same with
    the W_h_ and b_ terms.
    """
    def __init__(self, x_size=256, h_size=256, layer_norm=False,
                                               **kwargs):
        super(GRU, self).__init__()

        self.x_size = x_size
        self.h_size = h_size
        self.n_state_vars = 1

        # Internal GRU Entry Parameters
        means = torch.zeros(3, x_size, h_size)
        normal = torch.normal(means, std=1/float(np.sqrt(h_size)))
        self.W_x = nn.Parameter(normal, requires_grad=True)

        # Internal GRU State Parameters
        means = torch.zeros(3, h_size, h_size)
        normal = torch.normal(means, std=1/float(np.sqrt(h_size)))
        self.W_h = nn.Parameter(normal, requires_grad=True)

        # Internal GRU Bias Parameters
        self.b = nn.Parameter(torch.zeros(3,1,h_size),requires_grad=True)

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, old_h):
        """
        old_h - running state of GRU. FloatTensor Variable with shape
            (batch_size, h_size)
        x - New data coming into GRU. FloatTensor Variable with shape
            (batch_size, h_size)
        """
        z=self.sigmoid(x.mm(self.W_x[0])+old_h.mm(self.W_h[0])+self.b[0])
        r=self.sigmoid(x.mm(self.W_x[1])+old_h.mm(self.W_h[1])+self.b[1])
        h = z*old_h + (1-z)*self.tanh(x.mm(self.W_x[2]) +\
                                (r*old_h).mm(self.W_h[2]) + self.b[2])
        return h

    def check_grads(self):
        for p in list(self.parameters()):
            if torch.sum(p.grad.data != p.grad.data):
                print("NaN in grads")

class GRUFCModel(nn.Module):
    def __init__(self, input_shape, output_space, h_size=200,
                                                bnorm=False,
                                                is_discrete=True,
                                                **kwargs):
        super(GRUFCModel, self).__init__()

        self.h_size = h_size
        self.is_discrete = is_discrete
        self.is_recurrent = True
        self.flat_size = np.prod(input_shape[-3:])
        self.input_shape = input_shape
        self.output_space = output_space

        block = [nn.Linear(self.flat_size, h_size)]
        if bnorm:
            block.append(nn.BatchNorm1d(h_size))
        block.append(nn.ReLU())
        block.append(nn.Linear(h_size, h_size))
        if bnorm:
            block.append(nn.BatchNorm1d(h_size))

        self.base = nn.Sequential(*block)
        self.gru = GRU(x_size=self.h_size, h_size=self.h_size)

        outsize = output_space if is_discrete else 2*output_space
        self.action_out = nn.Linear(h_size, outsize)
        self.value_out = nn.Sequential(nn.LayerNorm(h_size),
                                       nn.Linear(h_size, 1),
                                       nn.Linear(1,1))

    def forward(self, x, old_h):
        fx = x.view(len(x), -1)
        fx = self.base(fx)
        h = self.gru(fx,old_h)
        pi = self.action_out(h)
        value = self.value_out(h)
        if not self.is_discrete:
            mu,sigma = torch.chunk(pi,2,dim=-1)
            sigma = F.softplus(sigma)+0.0001
            return value, (mu,sigma), h
        return value, pi, h

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of
        sneaking into pytorch...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print("NaNs in Grad!")

    def req_grads(self, grad_on):
        """
        Used to turn off and on all gradient calculation requirements
        for speed.

        grad_on - bool denoting whether gradients should be calculated
        """
        for p in self.parameters():
            p.requires_grad = grad_on


class GRUModel(nn.Module):
    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, input_space, output_space, h_size=288,
                                                  bnorm=False,
                                                  is_discrete=True,
                                                  **kwargs):
        super(GRUModel, self).__init__()

        self.is_recurrent = True
        self.input_space = input_space
        self.output_space = output_space
        self.h_size = h_size
        self.bnorm = bnorm
        self.is_discrete = is_discrete

        self.convs = nn.ModuleList([])

        # Embedding Net
        shape = input_space.copy()

        ksize=3; stride=1; padding=1; out_depth=16
        self.convs.append(self.conv_block(input_space[-3],out_depth,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                                     padding=padding,
                                                     stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=24
        self.convs.append(self.conv_block(in_depth,out_depth,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                                     padding=padding,
                                                     stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=32
        self.convs.append(self.conv_block(in_depth,out_depth,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                                     padding=padding,
                                                     stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=48
        self.convs.append(self.conv_block(in_depth,out_depth,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding,
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                                     padding=padding,
                                                     stride=stride)

        ksize=3; stride=2; padding=1; in_depth=out_depth
        out_depth=64
        self.convs.append(self.conv_block(in_depth,out_depth,
                                            ksize=ksize,
                                            stride=stride,
                                            padding=padding, 
                                            bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize,
                                                     padding=padding,
                                                     stride=stride)
        self.features = nn.Sequential(*self.convs)
        self.flat_size = int(np.prod(shape))
        print("Flat Features Size:", self.flat_size)
        lin = nn.Linear(self.flat_size, self.h_size)
        self.resize_emb = nn.Sequential(lin, nn.ReLU())

        # GRU Unit
        self.gru = GRU(x_size=self.h_size, h_size=self.h_size)

        # Policy
        outsize = self.output_space if self.is_discrete\
                                    else 2*self.output_space
        self.pi = nn.Linear(self.h_size, outsize)
        self.value = nn.Linear(self.h_size, 1)

    def get_new_shape(self, shape, depth, ksize, padding, stride):
        new_shape = [depth]
        for i in range(2):
            new_shape.append(self.new_size(shape[i+1], ksize, padding,
                                                              stride))
        return new_shape
        
    def new_size(self, shape, ksize, padding, stride):
        return (shape - ksize + 2*padding)//stride + 1

    def forward(self, x, old_h):
        embs = self.emb_net(x)
        h = self.gru(embs, old_h)
        val, pi = self.policy(h)
        return val, pi, h

    def emb_net(self, mdp_state):
        """
        Creates an embedding for the mdp_state.

        mdp_state - Variable FloatTensor with shape (B, C, H, W)
        """
        feats = self.features(mdp_state)
        feats = feats.view(feats.shape[0], -1)
        state_embs = self.resize_emb(feats)
        return state_embs

    def policy(self, h):
        """
        Uses the state embedding to produce an action.

        h - the state embedding created by the emb_net
        """
        pi = self.pi(h)
        value = self.value(h)
        if not self.is_discrete:
            mu,sigma = torch.chunk(pi,2,dim=-1)
            sigma = F.softplus(sigma)+0.0001
            return value, (mu,sigma)
        return value, pi

    def conv_block(self, chan_in, chan_out, ksize=3, stride=1,
                                                     padding=1,
                                                     activation="lerelu",
                                                     max_pool=False,
                                                     bnorm=True):
        block = []
        block.append(nn.Conv2d(chan_in, chan_out, ksize, stride=stride,
                                                      padding=padding))
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

    def req_grads(self, calc_bool):
        """
        An on-off switch for the requires_grad parameter for each
        internal Parameter.

        calc_bool - Boolean denoting whether gradients should be
                    calculated.
        """
        for param in self.parameters():
            param.requires_grad = calc_bool

