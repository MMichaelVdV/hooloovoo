from enum import Enum

import torch
import os


class Mode(Enum):
    INFERENCE = 0
    TRAINING = 1

    @staticmethod
    def from_string(txt):
        if txt == "INFERENCE":
            return Mode.INFERENCE
        elif txt == "TRAINING":
            return Mode.TRAINING
        else:
            raise ValueError("Invalid mode: " + txt)


def attempt_get_cuda_device(verbose=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            print("using cuda")
    else:
        device = torch.device('cpu')
        if verbose:
            print("using cpu")
    return device


def limit_cpu_usage_overkill(n_cores):
    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    torch.set_num_threads(n_cores)
    # torch.set_num_interop_threads(n_cores)
