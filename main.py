from hyperparams import HyperParams
from ppo import PPO
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    ppo_trainer = PPO()
    hyper_params = HyperParams()
    ppo_trainer.train(hyper_params.hyps)

