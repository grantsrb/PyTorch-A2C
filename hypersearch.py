from hyperparams import HyperParams, hyper_search, make_hyper_range
from a2c import A2C
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    a2c_trainer = A2C()
    hyps = dict()
    hyp_ranges = {
                "lambda_": [.94, .955, .97], 
                "lr": [5e-5, 1e-4, 2.5e-4],
                "val_const": [.001, .01, .1, 1],
                }
    keys = list(hyp_ranges.keys())
    hyps['gamma'] = .99
    hyps['entr_coef'] = .01
    hyps['env_type'] = "Breakout-v0"
    hyps['exp_name'] = "brkout"
    hyps['n_tsteps'] = 128
    hyps['n_rollouts'] = 11
    hyps['n_envs'] = 11
    hyps['max_tsteps'] = 5000000
    hyps['n_frame_stack'] = 3
    search_log = open(hyps['exp_name']+"_searchlog.txt", 'w')
    hyper_params = HyperParams(hyps)
    hyps = hyper_params.hyps

    hyper_search(hyper_params.hyps, hyp_ranges, keys, 0, a2c_trainer, search_log)
    search_log.close()

