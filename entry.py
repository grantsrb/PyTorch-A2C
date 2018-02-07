import sys
from collector import Collector
from updater import Updater
import dense_model as model
import torch
from torch.autograd import Variable
import gc
import resource

exp_name = 'test'
env_type = 'snake-v0'

# Hyperparameters
gamma = .99 # Reward discount factor
_lambda = .98 # GAE moving average factor
n_rollouts = 3 # Number of times to perform rollouts before updating model
n_envs = 16 # Number of environments
n_tsteps = 15 # Maximum number of steps to take in an environment for one episode
val_const = .1 # Scales the value portion of the loss function
entropy_const = 0.01 # Scales the entropy portion of the loss function
max_norm = 0.4 # Scales the gradients using their norm
lr = 1e-3 # Divide by batchsize as a shortcut to averaging the gradient over multiple batches
n_state_frames = 2 # number of observations to stack for a single environment state

# Environment Choices
grid_size = [15,15]
n_foods = 2
unit_size = 4
action_space = 4

test = False
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
        if "n_state_frames=" in str_arg: n_state_frames= int(str_arg[len('n_state_frames='):])
        if "env_type=" in str_arg: env_type = str_arg[len('env_type='):]

        if "exp_name=" in str_arg: exp_name= str_arg[len('exp_name='):]
        elif "test" in str_arg: test = True
        elif "resume" in str_arg: resume = True
        elif "render" in str_arg: render = True

print("Experiment Name:", exp_name)
print("Env Name:", env_type)
print("gamma:", gamma)
print("_lambda:", _lambda)
print("n_rollouts:", n_rollouts)
print("n_envs:", n_envs)
print("n_tsteps:", n_tsteps)
print("val_const:", val_const)
print("entropy_const:", entropy_const)
print("max_norm:", max_norm)
print("lr:", lr)
print("n_state_frames:", n_state_frames)
print("grid_size:", grid_size)
print("n_foods:", n_foods)
print("unit_size:", unit_size)
print("lr:", lr)
print("Test:", test)
print("Resume:", resume)
print("Render:", render)

if test:
    net_save_file = "test_net.p"
    optim_save_file = "test_optim.p"
    log_file = "test_log.txt"
else:
    net_save_file = exp_name+"_net.p"
    optim_save_file = exp_name+"_optim.p"
    log_file = exp_name+"_log.txt"


collector = Collector(n_envs=n_envs, grid_size=grid_size, n_foods=n_foods, unit_size=unit_size, n_state_frames=n_state_frames, net=None, n_tsteps=n_tsteps, gamma=gamma, env_type=env_type, preprocessor=model.Model.preprocess)
net = model.Model(collector.state_shape, action_space)
dummy = net.forward(Variable(torch.zeros(2,*collector.state_shape)))
collector.net = net
updater = Updater(collector.net, lr, entropy_const=entropy_const, value_const=val_const, gamma=gamma, _lambda=_lambda, max_norm=max_norm)

if resume:
    dummy = Variable(torch.ones(1,*collector.state_shape))
    updater.net.req_grads(False)
    updater.net.forward(dummy)
    updater.net.req_grads(True)
    updater.net.load_state_dict(torch.load(exp_name+'_net.p'))
    updater.optim.load_state_dict(torch.load(exp_name+'_optim.p'))
updater.optim.zero_grad()

epoch = 0
while True:
    epoch += 1
    print("Begin Epoch", epoch, "– T =", collector.T)
    for rollout in range(n_rollouts):
        data = collector.get_data(render)
        updater.calc_loss(*data)
    updater.update_model()
    updater.save_model(net_save_file, optim_save_file)
    updater.print_statistics()
    print("Average Reward:", collector.avg_reward)

    # Check for memory leaks
    gc.collect()
    max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))
