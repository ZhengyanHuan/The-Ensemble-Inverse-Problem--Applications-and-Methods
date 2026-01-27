import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
# import tqdm
# import tqdm.notebook as tn
# import matplotlib.pyplot as plt
# from cddpm import cddpm, denoiseNet
import configs
import numpy as np
import torchvision
# import ot
from sklearn.decomposition import PCA
import os



class myMNIST():
    def __init__(self, kernel_size=configs.kernel_size, sigma_min=configs.blur_min, sigma_max = configs.blur_max
                 , noise_std = configs.noise_std, device = configs.device, blur_again = False, loadtest = False, path_addition = ''):
        super().__init__()
        self.kernel_size = kernel_size
        self.blur_min = sigma_min
        self.blur_max = sigma_max
        self.noise_std = noise_std
        self.device = device
        self.blur_transform = configs.blur_transform_global
        # self.blur_transform = transforms.Compose([
        #     transforms.ToTensor(),  # Convert image to tensor
        #     transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=(self.blur_min, self.blur_max)),
        #     transforms.Lambda(lambda x: (x + torch.randn_like(x) * self.noise_std))
        #     # transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * self.noise_std, 0, 1))
        #     # Apply Gaussian Blur
        # ])

        # self.dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
        # self.dataset_blurred = MNIST('.', train=True, transform=self.blur_transform, download=False)
        dataset_path = path_addition + '../WFM/WassersteinFlowMatching/tutorials/train96.npy'
        blurred_dataset_path =path_addition + '../WFM/WassersteinFlowMatching/tutorials/blur96.npy'

        testset_path = '../WFM/WassersteinFlowMatching/tutorials/test96.npy'

        self.dataset = torch.Tensor(np.load(dataset_path)).to(self.device)
        if loadtest:
            self.testset = torch.Tensor(np.load(testset_path)).to(self.device)
            self.blurred_testset = configs.blur_transform_func(self.testset)
            self.testset_len = len(self.testset)
            print('testset loaded')
        else:
            self.testset = None
            self.blurred_testset = None
        
        self.dataset_len = len(self.dataset)

        if os.path.exists(blurred_dataset_path) and blur_again == False:
            self.blurred_dataset = torch.Tensor(np.load(blurred_dataset_path)).to(self.device)
            print('blurred dataset loaded')

        else:
            self.blurred_dataset = configs.blur_transform_func(self.dataset)
            np.save(blurred_dataset_path, self.blurred_dataset.cpu().numpy())
            print('blurred dataset saved')

        


    def get_mixed_data(self, batch_size, alpha = None, train = True):
        if alpha is None:
            alpha = np.random.rand()
            while not ((alpha>=0.1 and alpha<=0.4) or (alpha>=0.6 and alpha<=0.9)):
                alpha = np.random.rand()


        alpha_ind = int(alpha*100)
        # print(alpha_ind)
        if train:
            selected_indices = np.random.choice(self.dataset_len, batch_size, replace=False)
            # print(selected_indices)
            return self.dataset[selected_indices, alpha_ind:alpha_ind+1, :, :], self.blurred_dataset[selected_indices, alpha_ind:alpha_ind+1, :, :]
        else:
            selected_indices = np.random.choice(self.testset_len, batch_size, replace=False)
            # print(selected_indices)
            return self.testset[selected_indices, alpha_ind:alpha_ind + 1, :, :], self.blurred_testset[selected_indices,
                                                                                  alpha_ind:alpha_ind + 1, :, :]



