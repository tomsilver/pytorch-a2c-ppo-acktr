import torch
import os
import sys

from . import distributions, model, utils

# :'(
sys.modules['model'] = model
sys.modules['distributions'] = distributions
sys.modules['utils'] = utils

def load_ppo(load_dir, env_id):
    return torch.load(os.path.join(load_dir, env_id + ".pt"))
