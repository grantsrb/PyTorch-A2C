import torch
import a2c
import ml_utils
import torch.multiprocessing as mp

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    ml_utils.training.run_training(a2c.training.train)

