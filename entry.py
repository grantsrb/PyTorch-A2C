import sys
from collector import Collector
from updater import Updater
import torch
from torch.autograd import Variable
import numpy as np
import gc
import resource

# Change your policy file here!
import dense_model as model
print("Using dense_model as policy file.")

exp_name = 'default'
env_type = 'snake-v0'

# Hyperparameters
gamma = .99 # Reward discount factor
_lambda = .98 # GAE moving average factor
n_rollouts = 3 # Number of times to perform rollouts before updating model
n_envs = 15 # Number of environments
n_tsteps = 15 # Maximum number of steps to take in an environment for one episode
val_const = .5 # Scales the value portion of the loss function
entropy_const = 0.01 # Scales the entropy portion of the loss function
max_norm = 0.4 # Scales the gradients using their norm
lr = 1e-4/n_rollouts # Divide by batchsize as a shortcut to averaging the gradient over multiple batches
n_obs_stack = 2 # number of observations to stack for a single environment state
gae = True
reinforce = False
norm_advs = False # Boolean that will normalize the advantages if true
view_net_input = False # Boolean to watch the actual input to the neural net

# Environment Choices
grid_size = [15,15]
n_foods = 2
unit_size = 4

resume = False
render = False
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        str_arg = str(arg)
        if "gamma=" in str_arg: gamma = float(str_arg[len("gamma="):])
        if "_lambda=" in str_arg: _lambda = float(str_arg[len("_lambda="):])
        if "n_rollouts=" in str_arg: n_rollouts = int(str_arg[len("n_rollouts="):])
        if "n_envs=" in str_arg: n_envs = int(str_arg[len("n_envs="):])
        if "n_tsteps=" in str_arg: n_tsteps = int(str_arg[len("n_tsteps="):])
        if "val_const=" in str_arg: val_const = float(str_arg[len("val_const="):])
        if "entropy_const=" in str_arg: entropy_const = float(str_arg[len("entropy_const="):])
        if "max_norm=" in str_arg: max_norm = float(str_arg[len("max_norm="):])
        if "lr=" in str_arg: lr = float(str_arg[len("lr="):])
        if "grid_size=" in str_arg: grid_size= [int(str_arg[len('grid_size='):]),int(str_arg[len('grid_size='):])]
        if "n_foods=" in str_arg: n_foods= int(str_arg[len('n_foods='):])
        if "unit_size=" in str_arg: unit_size= int(str_arg[len('unit_size='):])
        if "n_obs_stack=" in str_arg: n_obs_stack= int(str_arg[len('n_obs_stack='):])
        if "env_type=" in str_arg: env_type = str_arg[len('env_type='):]

        if "exp_name=" in str_arg: exp_name= str_arg[len('exp_name='):]
        elif "resume=False" in str_arg: resume = False
        elif "resume" in str_arg: resume = True
        elif "render=False" in str_arg: render = False
        elif "render" in str_arg: render = True
        elif "gae=False" in str_arg: gae = False
        elif "gae=True" in str_arg: gae = True
        elif "reinforce=False" in str_arg: reinforce = False
        elif "reinforce=True" in str_arg: reinforce = True
        elif "norm_advs=False" in str_arg: norm_advs = False
        elif "norm_advs=True" in str_arg: norm_advs = True
        elif "view_net_input=False" in str_arg: view_net_input = False
        elif "view_net_input" in str_arg: view_net_input = True

if reinforce and gae:
    print("GAE will take precedence over REINFORCE in model updates.\nREINFORCE is effectively False.")

print("Experiment Name:", exp_name)
print("env_type:", env_type)
print("gamma:", gamma)
print("_lambda:", _lambda)
print("n_rollouts:", n_rollouts)
print("n_envs:", n_envs)
print("n_tsteps:", n_tsteps)
print("val_const:", val_const)
print("entropy_const:", entropy_const)
print("max_norm:", max_norm)
print("lr:", lr)
print("n_obs_stack:", n_obs_stack)
print("grid_size:", grid_size)
print("n_foods:", n_foods)
print("unit_size:", unit_size)
print("norm_advs:", norm_advs)
print("Resume:", resume)
print("Render:", render)
print("GAE:", gae)
print("REINFORCE:", reinforce)

net_save_file = exp_name+"_net.p"
optim_save_file = exp_name+"_optim.p"
log_file = exp_name+"_log.txt"

collector = Collector(n_envs=n_envs, grid_size=grid_size, n_foods=n_foods, unit_size=unit_size, n_obs_stack=n_obs_stack, net=None, n_tsteps=n_tsteps, gamma=gamma, env_type=env_type, preprocessor=model.Model.preprocess)
print("Obs Shape:,",collector.obs_shape)
print("Prep Shape:,",collector.prepped_shape)
print("State Shape:,",collector.state_shape)
net = model.Model(collector.state_shape, collector.action_space, env_type=env_type, view_net_input=view_net_input)
dummy = net.forward(Variable(torch.zeros(2,*collector.state_shape)))
collector.net = net
updater = Updater(collector.net, lr, entropy_const=entropy_const, value_const=val_const, gamma=gamma, _lambda=_lambda, max_norm=max_norm, norm_advs=norm_advs)

if resume:
    updater.net.load_state_dict(torch.load(exp_name+'_net.p'))
    updater.optim.load_state_dict(torch.load(exp_name+'_optim.p'))
updater.optim.zero_grad()

epoch = 0
while True:
    epoch += 1
    print("Begin Epoch", epoch, "– T =", collector.T)
    for rollout in range(n_rollouts):
        data = collector.get_data(render)
        updater.calc_loss(*data, gae, reinforce)
    updater.update_model()
    updater.save_model(net_save_file, optim_save_file)
    updater.print_statistics()
    print("Grad Norm:", updater.norm, "– Avg Action:", np.mean(data[3]))
    print("Average Reward:", collector.avg_reward, end='\n\n')

    # Check for memory leaks
    gc.collect()
    max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))
