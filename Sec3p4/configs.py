import numpy as np
import torch
import torchvision.transforms as transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 2000
total_steps = 1000
lr = 5e-5
batch_size = 64
save_every = 200

def beta(t):
    beta0 = 1e-4
    betaT = 0.02
    return beta0 + (t-1)*(betaT-beta0)/total_steps

def alpha(t):
    return 1-beta(t)

def bar_alpha(t):
    out = 1
    for i in range(t):
        out *= alpha(i)
    return out

def sigma(t):
    return np.sqrt(beta(t))


