from hyperparams import HyperParams, hyper_search, make_hyper_range
from a2c import A2C
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    a2c_trainer = A2C()
    hyps = dict()
    hyp_ranges = {
                'entr_coef': [.01, .005],
                'val_coef': [2, 1, .5],
                'lr': [2.5e-4, 1e-4, 8.5e-5],
                }
    keys = list(hyp_ranges.keys())
    hyps['lambda_'] = .96
    hyps['gamma'] = .99
    hyps['env_type'] = "Pong-v0"
    hyps['exp_name'] = "2pong"
    hyps['n_tsteps'] = 32
    hyps['n_rollouts'] = 32
    hyps['n_envs'] = 11
    hyps['max_tsteps'] = 5000000
    hyps['n_frame_stack'] = 2
    hyps['model_type'] = 'gru'
    hyps['optim_type'] = 'rmsprop'
    search_log = open(hyps['exp_name']+"_searchlog.txt", 'w')
    hyper_params = HyperParams(hyps)
    hyps = hyper_params.hyps

    hyper_search(hyper_params.hyps, hyp_ranges, keys, 0, a2c_trainer, search_log)
    search_log.close()

