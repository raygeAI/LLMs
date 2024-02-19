import torch
import random
import numpy as np
from transformers import set_seed


# set_random_seed 设置随机种子数
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)