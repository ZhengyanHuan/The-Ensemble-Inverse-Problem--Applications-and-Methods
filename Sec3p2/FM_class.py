import torch
import torch.nn as nn
import configs
from transformer import SetTransformer
import random
import tqdm
from dataset import generate_data

## MLP takes as input the noised-up data pair (x_t, y) at time-step t and returns a prediction of the noise at that time-step
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64):
        super().__init__()

        self.linear_model1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.01),
            nn.GELU()
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Dropout(0.01),
            nn.GELU()
        )
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
        x2 = self.linear_model2(x1 + self.embedding_layer(t.view(-1,1)))
        x = self.linear_model3(x2 + x1)
        return x
        # self.lstm = nn.LSTM()

class tFMModel(nn.Module):
    def __init__(self, device, input_dim, output_dim, batch_size, info_dim):
        '''
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : diffusion steps
        input_dim : dimension of data

        '''
        super().__init__()
        self.device = device

        self.mlp = MLP( 2*input_dim + info_dim, output_dim)

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
            x0 = torch.randn_like(x)
            # add noise up to a random timestep
            t = torch.rand( (len(x),)).to(device=self.device)

            # x_t is x with noise added
            x_t = (1-t.view(-1,1))*x0+t.view(-1,1)*x
            info_rep = self.dist_info(y.unsqueeze(0)).squeeze(0).expand(self.batch_size, -1)

        else:
            train = False
            #             t0=t # for debug
            x_t = x
            t = torch.tensor([t]).repeat(x.size(0)).to(self.device).to(torch.float32)
            info_rep = test_rep_info

        y = torch.cat((y, info_rep), dim=1)

        noised_pair = torch.cat((x_t, y), dim=1)

        # output is the prediction of the noise epsilon
        output = self.mlp(noised_pair, t)
        #         print(len(self.alpha_bars))
        #         if t0 == 1:
        #             print(output)

        return (output, x-x0) if train else output




class cFMModel(nn.Module):
    def __init__(self, device, input_dim, output_dim, batch_size):
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
        # self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        # mlp predicts the noise based on x_tilde, which is data with noise added
        self.mlp = MLP( input_dim , output_dim)

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
            x0 = torch.randn_like(x)
            # add noise up to a random timestep
            t = torch.rand( (len(x),)).to(device=self.device)
            x_t = (1-t.view(-1,1))*x0+t.view(-1,1)*x

        else:
            # train = False
            #             t0=t # for debug
            x_t = x
            t = torch.tensor([t]).repeat(x.size(0)).to(self.device).to(torch.float32)


        noised_pair = torch.cat((x_t, y), dim=1)

        # output is the prediction of the noise epsilon
        output = self.mlp(noised_pair, t)
        #         print(len(self.alpha_bars))
        #         if t0 == 1:
        #             print(output)

        return (output, x-x0) if train else output



class cFM():
    def __init__(self,cFM_name = configs.cFM_name, save_int = configs.default_save_int,
                 lr = configs.lr, device = configs.device, input_dim = configs.dim_data,
                 output_dim = configs.dim_data, batch_size = configs.batch_size,
                 have_rho = False):
        self.device = device
        self.cFM_name = cFM_name
        self.lr = lr
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
        self.cFM_model = cFMModel(self.device,  self.input_dim,
                      self.output_dim, self.batch_size)
        self.cFM_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.cFM_model.parameters(), lr=configs.lr)


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
            loss = self.cFM_model.loss_fn(x_batch_B2, y_batch_B2)
            loss.backward()
            self.optimizer.step()
            if (iteration+1) % 10000 == 0:
                print(loss)
            if iteration == 5 or (iteration+1) % self.save_int == 0:
                # save model state checkpoint

                torch.save(self.cFM_model.state_dict(), configs.ckpt_path + self.cFM_name +'_it_' + str(iteration+1) + '.pth')

            # pbar.update(10)
            # pbar.set_description(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")
            # pbar.set_description(f"train loss: {loss.item():.6f}")

    def sampling(self, sampling_number, y, samplingsteps = configs.T):
        '''
        sampling_number : number to generate
        y               : data to condition on
        '''

        x = torch.randn((sampling_number, self.output_dim)).to(device=self.device)
        for i in range(samplingsteps):
            # t = torch.tensor([i/samplingsteps]).repeat(x.size(0)).to(self.device).to(torch.float32)
            t = i/samplingsteps
            with torch.no_grad():
                predict_epsilon = self.cFM_model(x, y, t, train=False)
            x = x+predict_epsilon/samplingsteps


        return x

    def manual_save(self, name):
        torch.save(self.cFM_model.state_dict(), configs.ckpt_path +  str(name) + '.pth')

    def load_ckpt(self, path):
        self.cFM_model.load_state_dict(torch.load(path))


class tFM():
    def __init__(self, info_dim = configs.dim_info, tFM_name = configs.tFM_name, 
                 save_int = configs.default_save_int, lr = configs.lr, device = configs.device, 
                  input_dim = configs.dim_data, output_dim = configs.dim_data, batch_size = configs.batch_size):
        self.device = device
        self.info_dim = info_dim
        self.tFM_name = tFM_name
        self.lr = lr
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.save_int = save_int
        # self.info_dim = info_dim
        # device, beta_1, beta_T, T, input_dim, output_dim, batch_size, info_dim
        self.tFM_model = tFMModel(self.device, self.input_dim,
                      self.output_dim, self.batch_size, self.info_dim)
        self.tFM_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.tFM_model.parameters(), lr=configs.lr)



    def train(self, epoches):
        pbar = tqdm.tqdm(total=epoches, miniters=1000)
        for iteration in range(epoches):
            # lossavg = 0

            x_batch_B2, y_batch_B2, rho = generate_data(self.batch_size)
            x_batch_B2 = torch.tensor(x_batch_B2).to(self.device).to(torch.float32)
            y_batch_B2 = torch.tensor(y_batch_B2).to(self.device).to(torch.float32)


            self.optimizer.zero_grad()
            loss = self.tFM_model.loss_fn(x_batch_B2, y_batch_B2)
            loss.backward()
            self.optimizer.step()
            lossavg = loss.item()
            if iteration == 5 or (iteration+1) % self.save_int == 0:
                # save model state checkpoint
                torch.save(self.tFM_model.state_dict(), configs.ckpt_path + self.tFM_name +'_it_' + str(iteration+1) + '.pth')

            if (iteration+1)%500==0:
                pbar.update(500)
            # pbar.set_description(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")
                pbar.set_description(f"train loss: {lossavg:.6f}")



    def sampling(self, sampling_number, y, samplingsteps=configs.T):
        '''
        sampling_number : number to generate
        y               : data to condition on
        '''
        test_rep_info = self.tFM_model.dist_info(y.unsqueeze(0)).squeeze(0).expand(self.batch_size, -1)
        x = torch.randn((sampling_number, self.output_dim)).to(device=self.device)
        for i in range(samplingsteps):
            # t = torch.tensor([i / samplingsteps]).repeat(x.size(0)).to(self.device).to(torch.float32)
            t = i / samplingsteps
            with torch.no_grad():
                predict_epsilon = self.tFM_model(x, y, t, test_rep_info = test_rep_info)
            x = x + predict_epsilon / samplingsteps

        return x
    # call the reverse_step function to take away the prediction of the noise at each step


    def manual_save(self, name):
        torch.save(self.tFM_model.state_dict(), configs.ckpt_path + str(name) + '.pth')

    def load_ckpt(self, path):
        self.tFM_model.load_state_dict(torch.load(path))