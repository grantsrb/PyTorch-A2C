## Training Details
To train a model, run the following command:

`$ python3 main.py hyperparams.json hyperranges.json`

This will initiate a training session and train over all permutations of the hyperranges.json file. See the following section for understanding how to make your own json.

## Making a hyperparams json
The hyperparams json conatains all details about hyperparameters, architectures, and training specifications. The following keys can and must be specified.

```
    "exp_name":"name_of_experiment", # Enter a name here that the experiment will be saved to
    "seed":0, # Specify the random seed for the training for repeatability

    "model":"FCModel", # Use the name of a class from `models` to specify the model architecture
    "env_type":"Pong-v0", # specify the environment name here
    "optim_type":"RMSprop", # specify the optimizer class from the torch.optim package
    "max_tsteps": 1e7, # The maximum number of environment steps
    "n_tsteps": 15, # the number of steps per rollout
    "n_envs": 11, # the number of unique environments (a unique process will be spawned for each environment)
    "n_frame_stack":3, # the number of frames to stack for an observation (stacked along pixel dimension)
    "n_rollouts": 45, # The number of rollouts to collect for each update
    "n_past_rews": 25, # the number of rewards to include in the running average (display purposes only)
    "h_size":256, # The size of the hidden state if using a recurrent model
    "grid_size": 15, # The size of the grid if using snake environment
    "unit_size": 4, # The number of pixels per grid unit if using snake env
    "n_foods": 2, # number of food pieces visible at all times if using snake
    "lr":0.0001, # learning rate
    "lr_low": 1e-12, # the minimum possible learning rate
    "lambda_":0.98, # The exponential decay factor for GAE
    "gamma":0.99, # the reward discount factor
    "gamma_high":0.995, # the maximum discount factor
    "val_coef":0.5, # the portion of loss attributed to value predictions
    "entr_coef":0.005, # the portion of loss attributed to the entropy bonus
    "entr_coef_low":0.001, # the minimum possible entropy bonus factor
    "max_norm":0.5, # the maximum l2 norm of the gradient vector
    "render": false, # if true, visuals of the game will be rendered during training (drastically slows training)
    "decay_lr": false, # if true, the lr is decayed over the course of training
    "decay_entr": false, # if true, the entr_coef is decayed over the course of training
    "use_nstep_rets": false, # if true, n step returns are used (unstable, potentially bugged currently)
    "norm_advs": true, # if true, advantages are normalized
    "use_bnorm": false, # if true, model uses batch norm
    "use_bptt": false # if true, and model is recurrent, back prop through time is used
```
