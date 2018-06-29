from hyperparams import HyperParams
from a2c import A2C
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    a2c_trainer = A2C()
    hyps = dict()
    hyps['exp_name'] = "pong"
    hyps['env_type'] = "Pong-v0"
    hyps['model_type'] = 'gru'
    hyps['use_bptt'] = False
    hyps['entr_coef'] = .01
    hyps['entr_low'] = .001
    hyps['decay_entr'] = True
    hyps['val_coef'] = .5
    hyps['lr'] = 5e-4
    hyps['lr_low'] = 1e-10
    hyps['decay_lr'] = True
    hyps['gamma'] = .98
    hyps['lambda_'] = .95
    hyps['n_tsteps'] = 64
    hyps['n_rollouts'] = 32
    hyps['n_envs'] = 11
    hyps['max_tsteps'] = 40000000
    hyps['n_frame_stack'] = 3
    hyps['optim_type'] = 'rmsprop'
    hyper_params = HyperParams(hyps)
    a2c_trainer.train(hyper_params.hyps)

