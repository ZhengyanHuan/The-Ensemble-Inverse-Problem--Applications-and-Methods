#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


import configs
import tqdm
from torch.optim import Adam
from configs import alpha, bar_alpha, beta, sigma

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class denoiseNet(nn.Module):


    def __init__(self, channels=[32, 64, 128, 256], embed_dim=256, input_channels = 2):

        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        # self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t): #x:B2NN
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        # print(t.shape)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        # h = h / marginal_prob_std(t)[:, None, None, None]
        return h






class cddpm:
    def __init__(self, device = configs.device, epoch = configs.epoch, total_steps = configs.total_steps, lr = configs.lr, batch_size = configs.batch_size):
        super().__init__()
        self.epoch = epoch
        self.device = device
        self.denoise_model = torch.nn.DataParallel(denoiseNet()).to(self.device)
        self.total_steps = total_steps
        self.optimizer = Adam(self.denoise_model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.MSE = nn.MSELoss()


    def train(self, MNIST_dataset, model_type, save_name = 'test', rep_len = 8):
        # avg_loss = torch.tensor(-1.0).to(self.device)
        loss_record = np.array([])

        with tqdm.tqdm(total=self.epoch, desc=f"Epoch ", unit="batch") as pbar:
            for epoch_num in range(self.epoch):
                avg_loss = torch.tensor(0.0).to(self.device)
                num_items = 0
                # for (x_truth_B1NN,y1), (x_blurred_B1NN,y2) in zip(data_loader_truth,data_loader_blurred):
                for i in range(100*rep_len):

                    x_truth_B1NN, x_blurred_B1NN = MNIST_dataset.get_mixed_data( self.batch_size)

                    x_truth_B1NN = x_truth_B1NN.to(self.device)
                    x_blurred_B1NN = x_blurred_B1NN.to(self.device)

                    if model_type=='DDPM':
                        loss = self.loss_fn(self.denoise_model, x_truth_B1NN, x_blurred_B1NN)
                    else:
                        loss = self.loss_fn_FM(self.denoise_model, x_truth_B1NN, x_blurred_B1NN)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    avg_loss += loss
                    num_items += 1
                avg_loss = avg_loss/num_items

                loss_record = np.append(loss_record, avg_loss.item())
                pbar.set_postfix({
                    "Loss": f"{avg_loss:.4f}"
                })
                pbar.update(1)

                if (epoch_num + 1) % configs.save_every == 0 or epoch_num == 0:
                    torch.save(self.denoise_model.state_dict(),'./saved_model/' + save_name +'_'+str(epoch_num+1)+'.pth')
                    np.savez(
                        './saved_model/' + save_name + '_' + 'train_record.npz',
                        loss_record=loss_record)
        return self.denoise_model

    def loss_fn(self, model, x_truth_B1NN, x_blurred_B1NN):
        random_t = torch.randint(1, self.total_steps + 1, (1, x_truth_B1NN.shape[0]), device=self.device)[0, :]
        bar_alpha_list = torch.tensor([(bar_alpha(t)) for t in random_t], device=self.device)
        sqrt_bar_alpha_list = torch.sqrt(bar_alpha_list)
        sqrt_invbar_alpha_list = torch.sqrt(1 - bar_alpha_list)
        eps = torch.randn_like(x_truth_B1NN, device=self.device)

        xt_B1NN = (x_truth_B1NN * sqrt_bar_alpha_list[:, None, None, None]) + eps * sqrt_invbar_alpha_list[:, None, None, None]
        combinedx_B2NN = torch.cat([xt_B1NN, x_blurred_B1NN], dim=1)
        eps_est = model(combinedx_B2NN, random_t)

        loss = self.MSE(eps_est, eps)

        return loss

    def loss_fn_FM(self, model, x_truth_B1NN, x_blurred_B1NN):
        random_t = torch.rand(x_truth_B1NN.shape[0]).to(self.device)
        x_init_B1NN = torch.randn_like(x_truth_B1NN)
        v = x_truth_B1NN - x_init_B1NN
        random_t_mat = random_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        xt_B1NN = random_t_mat * x_truth_B1NN + (1 - random_t_mat) * x_init_B1NN
        combinedx_B2NN = torch.cat([xt_B1NN, x_blurred_B1NN], dim=1)
        u_est = model(combinedx_B2NN, random_t)

        loss = self.MSE(u_est, v)
        return loss

    def sampler(self, model, x_blurred_B1NN):
        batch_size = x_blurred_B1NN.shape[0]
        x_prev = torch.randn(batch_size, 1, 28, 28, device=self.device)
        for i in range(self.total_steps):
            t = (self.total_steps - i)
            if t > 1:
                z = torch.randn(batch_size, 1, 28, 28, device=self.device)
            else:
                z = torch.zeros(batch_size, 1, 28, 28, device=self.device)
            t_tensor = torch.ones(batch_size).to(self.device) * t
            with torch.no_grad():
                modelinp = torch.cat([x_prev, x_blurred_B1NN], dim=1)
                x = 1 / np.sqrt(alpha(t)) * (
                            x_prev - (1 - alpha(t)) / (np.sqrt(1 - bar_alpha(t))) * model(modelinp, t_tensor)) + sigma(
                    t) * z

            x_prev = x
        return x_prev

    def sampler_FM(self, model, x_blurred_B1NN, steps_num = None):
        if steps_num is None:
            steps_num = self.total_steps

        batch_size = x_blurred_B1NN.shape[0]
        x_prev = torch.randn_like(x_blurred_B1NN)
        for i in range(steps_num):
            t = i/steps_num
            t_tensor = torch.ones(batch_size).to(self.device) * t
            with torch.no_grad():
                modelinp = torch.cat([x_prev, x_blurred_B1NN], dim=1)
                v_est = model(modelinp, t_tensor)
            x_prev = x_prev + v_est / steps_num

        return x_prev