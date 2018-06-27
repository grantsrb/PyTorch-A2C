from hyperparams import HyperParams
from a2c import A2C
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    a2c_trainer = A2C()
    arg_hyps = dict()
    arg_hyps['lr'] = .0001
    arg_hyps['gamma'] = .99
    arg_hyps['lambda_'] = .98
    arg_hyps['val_coef'] = .5
    arg_hyps['entr_coef'] = .01
    arg_hyps['env_type'] = "Pong-v0"
    arg_hyps['exp_name'] = "testing"
    arg_hyps['n_tsteps'] = 32
    arg_hyps['n_rollouts'] = 32
    arg_hyps['n_envs'] = 11
    arg_hyps['max_tsteps'] = 1000000
    arg_hyps['n_frame_stack'] = 2
    arg_hyps['model_type'] = 'fc'
    hyper_params = HyperParams(arg_hyps)
    a2c_trainer.train(hyper_params.hyps)

