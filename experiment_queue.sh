#!/bin/bash

python3 entry.py exp_name=dense max_tsteps=20000000 model_type=dense n_obs_stack=2 use_bnorm=True max_norm=.5
python3 entry.py exp_name=conv max_tsteps=20000000 model_type=conv n_obs_stack=4 use_bnorm=False max_norm=.5
