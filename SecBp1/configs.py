import numpy as np
import torch
import torchvision.transforms as transforms
# from matplotlib.style import available

kernel_size = 5
blur_min = 1
blur_max = 3
available_num = [0]
noise_std = 0.1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 500
total_steps = 1000
lr = 5e-4
batch_size = 128
save_every = 500

top_k=20
info_embed_dim_out = 20

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


noise_std_global = 0.2 ** 0.5  # sqrt(0.2)
mask_prob = 0.9

blur_transform_global = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: torch.clip(x + torch.randn_like(x) * noise_std_global,0,1))
    transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_std_global),
    transforms.Lambda(lambda x: x * (torch.rand_like(x) > mask_prob)) #0.85
])

def blur_transform_func(original_tensor):
    return (original_tensor+ torch.randn_like(original_tensor) * noise_std_global) * (torch.rand_like(original_tensor) > mask_prob)


