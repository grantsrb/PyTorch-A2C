# A2C

## Description
This is an implementation of A2C written in PyTorch using OpenAI gym environments.

## Dependencies
- python 3.5 or later
- pip
- gym
- (gym-snake)[https://github.com/grantsrb/Gym-Snake]
- numpy
- matplotlib
- pytorch

## How to Use this Repo

### Jumping In
You probably know how to clone a repo if you're getting into RL. But in case you don't, open a bash terminal and type the command:

    $ git clone https://github.com/grantsrb/PyTorch-A2C

Then navigate to the top folder using

    $ cd PyTorch-A2C

Hopefully you have already installed all the appropriate dependencies. See the section called **Dependencies** for a list of required dependencies.

From the top directory, the code can be immediately run on either the Pong-v0 or snake-v0 environments with good results. Simply run the command:

    $ python entry.py env_type=Pong-v0

or

    $ python entry.py env_type=snake-v0

This will run a training session with the name `default`. As such, it will save the trained net's state dict to a file called `default_net.p` and the optimizer's state dict to a file called `default_optim.p`. If you stop training and would like to resume from where you left off, simply add the argument `resume` to the command line arguments. You can change the name of the session from default by passing the command line argument `exp_name=<some_name_here>` which will save the net and optimizer state dicts to the names `some_name_here_net.p` and `some_name_here_optim.p` respectively.

### Watching Your Trained Policy
After training your policy, you can watch the policy in action using the `watch_model.py` file. To use this file, pass the name of the PyTorch state dict that you would like to view as the first command line argument and then specify the environment type using `env_type=<name_of_gym_environment>`. Make sure you are using the correct policy file by changing the 11th line of code from the current file to the file your state dict was created from. This will not be necessary if you did not change the default policy.

### Command Line Arguments
RL algorithms need tuning to be good. There are tons of hyperparameters that can potentially have a huge impact on the training of the algorithm. In order to help with automated hyperparameter tuning, this project offers a number of optional command line arguments. Each is set using `<argument_name>=<argument>` with no spaces. For example, if you wanted to set the variable n_envs (the number of environments) to 15, then you would use the following:

  $ python entry.py n_envs=15

Here is a detailed list of each parameter and what it does.

#### General
* `env_type` - string of the type of environment you would like to use A2C on. The environment must be an OpenAI gym environment.
* `exp_name` - string of the name of the experiment. Determines the name that the PyTorch state dicts are saved to.
* `gamma` - float value of the discount factor used to discount the rewards and advantages.
* `_lambda` - float value of the generalized advantage estimation moving average factor. Only applies if using GAE.
* `n_envs` - integer number of separate environments to instantiate and use for training.
* `n_tsteps` - integer number of steps to perform in each environment per episode
* `n_rollouts` - integer number of episodes to perform per gradient descent update on policy
* `val_const` - float value determining weight of the value loss in the total loss calculation
* `entropy_const` - float value determining weight of the entropy in the total loss calculation
* `max_norm` - float value denoting maximum gradient norm for gradient norm clipping
* `lr` - float value denoting the learning rate
* `n_obs_stack` - integer number denoting number of observations to stack to be used as the environment state. Must be 2 if using dense_model as your policy.
* `resume` - boolean denoting whether the training should be resumed from a previous point.
* `render` - boolean denoting whether the gym environment should be rendered
* `gae` - boolean denoting whether generalized advantage estimation should be used during training.
* `reinforce` - boolean denoting whether vanilla REINFORCE type updates should be used for training. If gae and reinforce are both marked true, gae takes precedence.
* `norm_advs` - boolean denoting whether the advantages should be normalized prior to multiplication with the action log probabilities.
* `view_net_input` - boolean to view the actual inputs to the policy net. Can only be used when using dense_model.

#### Specific to snake-v0
* `grid_size` - integer denoting square dimensions for size of grid for snake.
* `n_foods` - integer denoting number of food pieces to appear on grid
* `unit_size` - integer denoting number of pixels per unit in grid.

### Using New Environments
In order to use a new environment, all that needs to be done is pass the name of the new environment using the `env_type=` command line argument. The environment must be an OpenAI gym environment.

Ideally you should add some preprocessing to the static method `preprocess` in the file that your `Model` is defined. See **Using Different Policies** to see more details.

### Using Different Policies
In order to use a different policy, use a python file with that implements a class called `Model` that is a subclass of PyTorch's `nn.Module` class. Then change the 11th line of code in `entry.py` to import the file containing your policy as `model`.

The `Model` class must have the following methods:

- __init__(self, input_space, output_space) where input_space is a tuple denoting the shape of the input to the net and the output space is an integer denoting the number of possible actions.
- forward(self, x) where x is a torch.autograd.Variable
- req_grads(self, requires_grad_boolean) which is a function used for efficiency and can simply be copied and pasted from the code below:

  def req_grads(self, grad_on):
    for p in self.parameters():
        p.requires_grad = grad_on

- preprocess(pic) which is a static method that should do any preprocessing to the observations from the environment before the observations are stacked into the state. The return type should be (1, prepped_obs_shape) so that the observation can be concatenated with the previous state.
