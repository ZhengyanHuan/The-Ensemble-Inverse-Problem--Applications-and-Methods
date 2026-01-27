import torch
import torch.nn as nn
import configs

import numpy as np


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


## MLP takes as input the noised-up data pair (x_t, y) at time-step t and returns a prediction of the noise at that time-step
class MLP(nn.Module):
    def __init__(self, n_steps, input_dim, output_dim):
        super().__init__()

        self.linear_model1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(0.01),
            nn.GELU()
        )
        # self.embedding_layer = nn.Embedding(n_steps, 256)
        self.embedding_layer = nn.Sequential(
            GaussianFourierProjection(embed_dim=256),
            nn.Linear(256, 256)
        )

        self.linear_model2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(0.01),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(0.01),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(0.01),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(0.01),
            nn.GELU(),

            nn.Linear(512, 256),
        )

        self.linear_model3 = nn.Sequential(
            nn.Linear(256, output_dim)
        )

    def forward(self, x, t):
        x1 = self.linear_model1(x)
        x2 = self.linear_model2(x1 + self.embedding_layer(t))
        x = self.linear_model3(x2 + x1)
        return x
        # self.lstm = nn.LSTM()




class Model_vanilla(nn.Module):
    def __init__(self, device, beta_1, beta_T, T, input_dim, output_dim, batch_size):
        '''
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : diffusion steps
        input_dim : dimension of data

        '''
        super().__init__()
        self.device = device
        # beta's are the schedule with which the noise is added to the data
        # alpha_bars is cumulative product of alpha_bar (1 - beta_1) * (1 - beta_2) * ... * (1 - beta_T)
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        # mlp predicts the noise based on x_tilde, which is data with noise added
        self.mlp = MLP(T, input_dim , output_dim)


        self.batch_size = batch_size
        self.to(device=self.device)

    def loss_fn(self, x, y, t=None, model_type="DDPM"):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if None (training phase), we add noise at random timesteps.
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.

        '''
        # output comes from forward process, value calculated in backbone
        # epsilon is a sampled random parameter that determines the noise added onto the data (vector of random numbers shaped like data)
        # alpha's are like normalization factors that are used to scale the noise

        if model_type == "DDPM":
            output, epsilon = self.forward(x, y, t=t, train=True)
        elif model_type == "FM":
            output, epsilon = self.forward_FM(x, y, t=t, train=True)
        else:
            raise ValueError("model_type must be DDPM or FM")
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, y, t, train):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if training phase, we add noise at random timesteps.
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.
        train      : if True (training phase), target is returned along with epsilon prediction

        '''

        if train == True:
            epsilon = torch.randn_like(x)
            # add noise up to a random timestep
            t = torch.randint(0, len(self.alpha_bars), (len(x),)).to(device=self.device)
            # used_alpha_bar is the cumulative product of alpha's used up to time t in this iteration
            used_alpha_bars = self.alpha_bars[t][:, None]

            # x_t is x with noise added
            x_t = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
            # info_rep = self.dist_info(y.unsqueeze(0)).squeeze(0).expand(self.batch_size, -1)

        else:
            #             t0=t # for debug
            x_t = x
            t = torch.tensor([t]).repeat(x.size(0)).to(self.device).long()

        noised_pair = torch.cat((x_t, y), dim=1)
        # output is the prediction of the noise epsilon
        output = self.mlp(noised_pair, t)
        #         print(len(self.alpha_bars))
        #         if t0 == 1:
        #             print(output)

        return (output, epsilon) if train else output

    def forward_FM(self, x, y, t, train):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if training phase, we add noise at random timesteps.
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.
        train      : if True (training phase), target is returned along with epsilon prediction

        '''

        if train == True:
            x0 = torch.randn_like(x)
            # add noise up to a random timestep
            t = torch.rand(x.shape[0]).to(device=self.device)
            t_mat = t.unsqueeze(-1)
            x_t = t_mat * x + (1 - t_mat) * x0

        else:

            #             t0=t # for debug
            x_t = x
            t = torch.tensor([t], dtype=torch.float32).repeat(x.size(0)).to(self.device)
        noised_pair = torch.cat((x_t, y), dim=1)

        output = self.mlp(noised_pair, t)
        #         print(len(self.alpha_bars))
        #         if t0 == 1:
        #             print(output)

        return (output, x - x0) if train else output
