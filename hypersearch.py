from hyperparams import HyperParams, hyper_search, make_hyper_range
from a2c import A2C
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    a2c_trainer = A2C()
    hyps = dict()
    hyp_ranges = {
                'lr': [5e-4, 2.5e-4, 1e-4],
                'gamma': [.96, .98, .99],
                'lambda_':[.94, .95, .97, .98]
                }
    keys = list(hyp_ranges.keys())
    hyps['entr_coef'] = .01
    hyps['val_coef'] = .5
    hyps['env_type'] = "Pong-v0"
    hyps['exp_name'] = "4pong"
    hyps['n_tsteps'] = 32
    hyps['n_rollouts'] = 32
    hyps['n_envs'] = 11
    hyps['max_tsteps'] = 5000000
    hyps['n_frame_stack'] = 3
    hyps['model_type'] = 'gru'
    hyps['optim_type'] = 'rmsprop'
    search_log = open(hyps['exp_name']+"_searchlog.txt", 'w')
    hyper_params = HyperParams(hyps)
    hyps = hyper_params.hyps

    hyper_search(hyper_params.hyps, hyp_ranges, keys, 0, a2c_trainer, search_log)
    search_log.close()

