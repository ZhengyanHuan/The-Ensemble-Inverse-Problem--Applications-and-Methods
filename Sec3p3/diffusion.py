import torch
import configs
class FMProcess():
    def __init__(self,T, FM_fn, device, shape, dist_info_required = True):
        self.FM_fn = FM_fn
        self.device = device
        self.T = T
        self.shape = shape
        self.dist_info_required = dist_info_required

    def sampling(self, sampling_number, y ):
        if self.dist_info_required:
            test_rep_info = self.FM_fn.dist_info(y.unsqueeze(0)).squeeze(0).expand(self.FM_fn.batch_size, -1)
        x_prev = torch.randn((sampling_number, *self.shape)).to(device=self.device)
        for i in range(self.T):
            t = i / self.T
            with torch.no_grad():
                if self.dist_info_required:
                    predict_v = self.FM_fn.forward_FM(x_prev, y, t, test_rep_info=test_rep_info)
                else:
                    predict_v = self.FM_fn.forward_FM(x_prev, y, t, train = False)
                # print(y)
            x_prev = x_prev + predict_v / self.T
        return x_prev

class DiffusionProcess():
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape, dist_info_required = True):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''

        self.betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start = beta_1, end=beta_T, steps=T), dim = 0).to(device = device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape
        
        self.diffusion_fn = diffusion_fn
        self.device = device
        self.dist_info_required = dist_info_required


    # does the reverse process for diffusion, yields a predicted value x at each step
    # x is the de-noised prediction of the data at each step
    def reverse_step(self, x, y):
        '''
        x   : perturbated data
        y   : data to condition on
        '''
        if self.dist_info_required:
            test_rep_info = self.diffusion_fn.dist_info(y.unsqueeze(0)).squeeze(0).expand(self.diffusion_fn.batch_size, -1)
        for t in reversed(range(len(self.alpha_bars))):

            # use the trained diffusion network to predict the FULL amount noise that was added by that timestep in the forward process
            if self.dist_info_required:
                predict_epsilon = self.diffusion_fn(x, y, t, test_rep_info = test_rep_info)
            else:
                predict_epsilon = self.diffusion_fn(x, y, t, train = False)
#             print(predict_epsilon)
            # mu_theta_xt is X minus the predicted noise subtracted for ALL timesteps
            mu_theta_xt = torch.sqrt(1 / self.alphas[t]) * (x - self.betas[t] / torch.sqrt(1 - self.alpha_bars[t]) * predict_epsilon)

            # since we want to go in steps (not all at once), we have to add back noise for the timestep we are at according to the noise schedule
            # so create some noise according to the noising schedule (sqrt_tilde_beta * noise)
            noise = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[t]) / (1 - self.alpha_bars[t]) * self.betas[t])
            sqrt_beta = torch.sqrt(self.betas[t])
            sigma_t = sqrt_beta # or sqrt_tilde_beta
            # add this noise to mu_theta_xt, which was the data minus the predicted noise subtracted for ALL timesteps
            # x is then the prediction of the data at each timestep
            x = mu_theta_xt + sigma_t * noise
            yield x
    

    @torch.no_grad()
    # call the reverse_step function to take away the prediction of the noise at each step
    def sampling(self, sampling_number, y):
        '''
        sampling_number : number to generate
        y               : data to condition on
        '''

        sample_noise = torch.randn((sampling_number, *self.shape)).to(device = self.device)
        
        final = None
        for t, sample in enumerate(self.reverse_step(sample_noise, y)):
            final = sample

        return final