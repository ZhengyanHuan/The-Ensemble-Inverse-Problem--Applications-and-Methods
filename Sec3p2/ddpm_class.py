import torch
import torch.nn as nn
import configs
from transformer import SetTransformer
import random
import tqdm
from dataset import generate_data

## MLP takes as input the noised-up data pair (x_t, y) at time-step t and returns a prediction of the noise at that time-step
class MLP(nn.Module):
    def __init__(self, n_steps, input_dim, output_dim, hidden_dim = 64):
        super().__init__()

        self.linear_model1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.01),
            nn.GELU()
        )
        self.embedding_layer = nn.Embedding(n_steps, hidden_dim)

        self.linear_model2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.01),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.01),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.01),
            nn.GELU(),

            # nn.Linear(512, 512),
            # nn.Dropout(0.01),
            # nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
        )

        self.linear_model3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        x1 = self.linear_model1(x)
        x2 = self.linear_model2(x1 + self.embedding_layer(t))
        x = self.linear_model3(x2 + x1)
        return x
        # self.lstm = nn.LSTM()

class tddpmModel(nn.Module):
    def __init__(self, device, beta_1, beta_T, T, input_dim, output_dim, batch_size, info_dim):
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
        self.mlp = MLP(T, 2*input_dim + info_dim, output_dim)

        self.dist_info = SetTransformer(input_dim, 1, info_dim, 16, 32)

        self.batch_size = batch_size
        self.to(device=self.device)

    def loss_fn(self, x, y, t=None):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if None (training phase), we add noise at random timesteps.
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.

        '''
        # output comes from forward process, value calculated in backbone
        # epsilon is a sampled random parameter that determines the noise added onto the data (vector of random numbers shaped like data)
        # alpha's are like normalization factors that are used to scale the noise

        output, epsilon = self.forward(x, y, t=t)
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, y, t, test_rep_info=None):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if training phase, we add noise at random timesteps.
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.
        train      : if True (training phase), target is returned along with epsilon prediction

        '''

        if test_rep_info == None:
            train = True
            epsilon = torch.randn_like(x)
            # add noise up to a random timestep
            t = torch.randint(0, len(self.alpha_bars), (len(x),)).to(device=self.device)
            # used_alpha_bar is the cumulative product of alpha's used up to time t in this iteration
            used_alpha_bars = self.alpha_bars[t][:, None]

            # x_t is x with noise added
            x_t = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
            info_rep = self.dist_info(y.unsqueeze(0)).squeeze(0).expand(self.batch_size, -1)

        else:
            train = False
            #             t0=t # for debug
            x_t = x
            t = torch.tensor([t]).repeat(x.size(0)).to(self.device).long()
            info_rep = test_rep_info

        #         print(info_rep.shape)

        ######################################
        # The output of the deep set is shape [batch_size, info_dim],
        # the output of set transformer is [batch_size,num_outputs, info_dim],num_outputs is recommended to set to 1 for most cases,
        # unless  for problems such as amortized clustering which requires k correlated outputs

        # batch size if 1 in our cddpm,  ".squeeze(0).expand(self.batch_size, -1)" can work for both

        ########################################

        y = torch.cat((y, info_rep), dim=1)

        noised_pair = torch.cat((x_t, y), dim=1)

        # output is the prediction of the noise epsilon
        output = self.mlp(noised_pair, t)
        #         print(len(self.alpha_bars))
        #         if t0 == 1:
        #             print(output)

        return (output, epsilon) if train else output




class cddpmModel(nn.Module):
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

    def loss_fn(self, x, y, t=None):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if None (training phase), we add noise at random timesteps.
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.

        '''
        # output comes from forward process, value calculated in backbone
        # epsilon is a sampled random parameter that determines the noise added onto the data (vector of random numbers shaped like data)
        # alpha's are like normalization factors that are used to scale the noise

        output, epsilon = self.forward(x, y, t=t)
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, y, t, train=True):
        '''
        x          : truth-data if train=True, else its generated noise
        y          : reco-data, don't want to add noise to this
        t          : if training phase, we add noise at random timesteps.
                   : else (inference phase), we predict noise at specified sequence of timesteps from t=T -> t=0.
        train      : if True (training phase), target is returned along with epsilon prediction

        '''

        if train == True:
            # train = True
            epsilon = torch.randn_like(x)
            # add noise up to a random timestep
            t = torch.randint(0, len(self.alpha_bars), (len(x),)).to(device=self.device)
            # used_alpha_bar is the cumulative product of alpha's used up to time t in this iteration
            used_alpha_bars = self.alpha_bars[t][:, None]

            # x_t is x with noise added
            x_t = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon


        else:
            # train = False
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



class cDDPM():
    def __init__(self,cddpm_name = configs.cddpm_name, save_int = configs.default_save_int, lr = configs.lr, device = configs.device, beta_1 = configs.beta_1, beta_T = configs.beta_T, T = configs.T, input_dim = configs.dim_data, output_dim = configs.dim_data, batch_size = configs.batch_size,
                 have_rho = False):
        self.device = device
        self.cddpm_name = cddpm_name
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.have_rho = have_rho
        if have_rho:
            self.input_dim = 2*input_dim +1
        else:
            self.input_dim = 2*input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.save_int = save_int
        # self.info_dim = info_dim
        # device, beta_1, beta_T, T, input_dim, output_dim, batch_size, info_dim
        self.cDDPM_model = cddpmModel(self.device, self.beta_1, self.beta_T, self.T, self.input_dim,
                      self.output_dim, self.batch_size)
        self.cDDPM_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.cDDPM_model.parameters(), lr=configs.lr)

        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])


    def train(self, epoches):
        # pbar = tqdm.tqdm(total=epoches)
        for iteration in tqdm.tqdm(range(epoches), miniters=500):
            x_batch_B2, y_batch_B2, rho = generate_data(self.batch_size)
            x_batch_B2 = torch.tensor(x_batch_B2).to(self.device).to(torch.float32)
            y_batch_B2 = torch.tensor(y_batch_B2).to(self.device).to(torch.float32)
            if self.have_rho:
                rho_tensor = torch.full((y_batch_B2.shape[0], 1), rho, dtype=torch.float32, device=self.device)
                y_batch_B2 = torch.cat([y_batch_B2, rho_tensor], dim=1)

            self.optimizer.zero_grad()
            loss = self.cDDPM_model.loss_fn(x_batch_B2, y_batch_B2)
            loss.backward()
            self.optimizer.step()
            if (iteration+1) % 10000 == 0:
                print(loss)
            if iteration == 5 or (iteration+1) % self.save_int == 0:
                # save model state checkpoint

                torch.save(self.cDDPM_model.state_dict(), configs.ckpt_path + self.cddpm_name +'_it_' + str(iteration+1) + '.pth')

            # pbar.update(10)
            # pbar.set_description(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")
            # pbar.set_description(f"train loss: {loss.item():.6f}")

    def reverse_step(self, x, y):
        '''
        x   : perturbated data
        y   : data to condition on
        '''
        # test_rep_info = self.diffusion_fn.dist_info(y.unsqueeze(0)).squeeze(0).expand(self.diffusion_fn.batch_size, -1)
        for t in reversed(range(len(self.alpha_bars))):
            # use the trained diffusion network to predict the FULL amount noise that was added by that timestep in the forward process

            predict_epsilon = self.cDDPM_model(x, y, t, train = False)

            mu_theta_xt = torch.sqrt(1 / self.alphas[t]) * (
                        x - self.betas[t] / torch.sqrt(1 - self.alpha_bars[t]) * predict_epsilon)


            noise = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
            # sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[t]) / (1 - self.alpha_bars[t]) * self.betas[t])
            sqrt_beta = torch.sqrt(self.betas[t])
            sigma_t = sqrt_beta  # or sqrt_tilde_beta

            x = mu_theta_xt + sigma_t * noise
            yield x

    @torch.no_grad()
    # call the reverse_step function to take away the prediction of the noise at each step
    def sampling(self, sampling_number, y):
        '''
        sampling_number : number to generate
        y               : data to condition on
        '''

        sample_noise = torch.randn((sampling_number, self.output_dim)).to(device=self.device)

        final = None
        for t, sample in enumerate(self.reverse_step(sample_noise, y)):
            final = sample

        return final

    def manual_save(self, name):
        torch.save(self.cDDPM_model.state_dict(), configs.ckpt_path +  str(name) + '.pth')

    def load_ckpt(self, path):
        self.cDDPM_model.load_state_dict(torch.load(path))


class tDDPM():
    def __init__(self, info_dim = configs.dim_info, tddpm_name = configs.tddpm_name, save_int = configs.default_save_int, lr = configs.lr, device = configs.device, beta_1 = configs.beta_1, beta_T = configs.beta_T, T = configs.T, input_dim = configs.dim_data, output_dim = configs.dim_data, batch_size = configs.batch_size):
        self.device = device
        self.info_dim = info_dim
        self.tddpm_name = tddpm_name
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.save_int = save_int
        # self.info_dim = info_dim
        # device, beta_1, beta_T, T, input_dim, output_dim, batch_size, info_dim
        self.tDDPM_model = tddpmModel(self.device, self.beta_1, self.beta_T, self.T, self.input_dim,
                      self.output_dim, self.batch_size, self.info_dim)
        self.tDDPM_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.tDDPM_model.parameters(), lr=configs.lr)

        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])


    def train(self, epoches):
        pbar = tqdm.tqdm(total=epoches)
        for iteration in range(epoches):
            # lossavg = 0

            x_batch_B2, y_batch_B2, rho = generate_data(self.batch_size)
            x_batch_B2 = torch.tensor(x_batch_B2).to(self.device).to(torch.float32)
            y_batch_B2 = torch.tensor(y_batch_B2).to(self.device).to(torch.float32)


            self.optimizer.zero_grad()
            loss = self.tDDPM_model.loss_fn(x_batch_B2, y_batch_B2)
            loss.backward()
            self.optimizer.step()
            lossavg = loss.item()
            if iteration == 5 or (iteration+1) % self.save_int == 0:
                # save model state checkpoint
                torch.save(self.tDDPM_model.state_dict(), configs.ckpt_path + self.tddpm_name +'_it_' + str(iteration+1) + '.pth')

            if (iteration+1)%500==0:
                pbar.update(500)
            # pbar.set_description(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")
                pbar.set_description(f"train loss: {lossavg:.6f}")

    def reverse_step(self, x, y):
        '''
        x   : perturbated data
        y   : data to condition on
        '''
        test_rep_info = self.tDDPM_model.dist_info(y.unsqueeze(0)).squeeze(0).expand(self.batch_size, -1)
        for t in reversed(range(len(self.alpha_bars))):
            # use the trained diffusion network to predict the FULL amount noise that was added by that timestep in the forward process

            predict_epsilon = self.tDDPM_model(x, y, t, test_rep_info = test_rep_info)

            mu_theta_xt = torch.sqrt(1 / self.alphas[t]) * (
                        x - self.betas[t] / torch.sqrt(1 - self.alpha_bars[t]) * predict_epsilon)


            noise = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
            # sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[t]) / (1 - self.alpha_bars[t]) * self.betas[t])
            sqrt_beta = torch.sqrt(self.betas[t])
            sigma_t = sqrt_beta  # or sqrt_tilde_beta

            x = mu_theta_xt + sigma_t * noise
            yield x

    @torch.no_grad()
    # call the reverse_step function to take away the prediction of the noise at each step
    def sampling(self, sampling_number, y):
        '''
        sampling_number : number to generate
        y               : data to condition on
        '''

        sample_noise = torch.randn((sampling_number, self.output_dim)).to(device=self.device)

        final = None
        for t, sample in enumerate(self.reverse_step(sample_noise, y)):
            final = sample

        return final

    def manual_save(self, name):
        torch.save(self.tDDPM_model.state_dict(), configs.ckpt_path + str(name) + '.pth')

    def load_ckpt(self, path):
        self.tDDPM_model.load_state_dict(torch.load(path))