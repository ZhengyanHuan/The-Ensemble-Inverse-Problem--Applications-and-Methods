import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import configs
from torch.optim import Adam
import numpy as np
import tqdm
from InversionNet import SupervisedNet

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)

        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)

        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)

        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)

        # 2×2 kernel collapses 2×2 → 1×1
        self.convblock5 = nn.Conv2d(dim4, 1, kernel_size=2, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)

        x = self.convblock2_1(x)
        x = self.convblock2_2(x)

        x = self.convblock3_1(x)
        x = self.convblock3_2(x)

        x = self.convblock4_1(x)
        x = self.convblock4_2(x)

        x = self.convblock5(x)   # (B, 1, 1, 1)
        return x.view(x.size(0), -1)  # (B, 1)



class Wasserstein_GP(nn.Module):
    def __init__(self, device, lambda_gp):
        super(Wasserstein_GP, self).__init__()
        self.device = device
        self.lambda_gp = lambda_gp

    def forward(self, real, fake, model):
        gradient_penalty = self.compute_gradient_penalty(model, real, fake)
        loss_real = torch.mean(model(real))
        loss_fake = torch.mean(model(fake))
        loss = -loss_real + loss_fake + gradient_penalty * self.lambda_gp
        return loss, loss_real-loss_fake, gradient_penalty

    def compute_gradient_penalty(self, model, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = model(interpolates)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(real_samples.size(0), d_interpolates.size(1)).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty




class Gan:
    def __init__(self, device = configs.device, epoch = configs.epoch, lr = configs.lr, batch_size = configs.batch_size):
        super().__init__()
        self.epoch = epoch
        self.device = device

        self.model = SupervisedNet().to(self.device)
        self.model_d = Discriminator().to(self.device)
        # self.total_steps = total_steps
        self.optimizer_d = Adam(self.model_d.parameters(), lr=lr)
        self.optimizer_g = Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.n_critic = 5
        self.lambda_gp = 10.0
        self.criterion_d = Wasserstein_GP(self.device, self.lambda_gp).to(self.device)
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def criterion_g(self, pred, gt, model_d=None, lambda_g1v=100, lambda_g2v=100, lambda_adv=1):
        loss_g1v = self.l1loss(pred, gt)
        loss_g2v = self.l2loss(pred, gt)
        loss = lambda_g1v * loss_g1v + lambda_g2v * loss_g2v
        if model_d is not None:
            loss_adv = -torch.mean(model_d(pred))
            loss += lambda_adv * loss_adv
        return loss, loss_g1v, loss_g2v

    def train(self, MNIST_dataset, save_name='test', init_ckpt=None, rep_len=8, exclude=None, init_ckpt_d = None):
        # avg_loss = torch.tensor(-1.0).to(self.device)
        loss_record_d = np.array([])
        loss_record_g = np.array([])
        if init_ckpt is not None:
            self.model.load_state_dict(init_ckpt)
        if init_ckpt_d is not None:
            self.model_d.load_state_dict(init_ckpt_d)


        with tqdm.tqdm(total=self.epoch, desc=f"Epoch ", unit="batch") as pbar:
            for epoch_num in range(self.epoch):
                avg_loss_d = torch.tensor(0.0).to(self.device)
                avg_loss_g = torch.tensor(0.0).to(self.device)
                num_items_d = 0
                num_items_g = 0
                # for (x_truth_B1NN,y1), (x_blurred_B1NN,y2) in zip(data_loader_truth,data_loader_blurred):
                for i in range(100*rep_len):

                    x_truth_B1NN, x_blurred_B1NN = MNIST_dataset.get_mixed_data( self.batch_size, exclude = exclude)

                    self.optimizer_d.zero_grad()
                    with torch.no_grad():
                        pred = self.model(x_blurred_B1NN)
                    loss_d, loss_diff, loss_gp = self.criterion_d(x_truth_B1NN, pred, self.model_d)
                    loss_d.backward()
                    self.optimizer_d.step()
                    avg_loss_d += loss_d.item()
                    num_items_d += 1

                    if ((i + 1) % self.n_critic == 0) or (i == 100*rep_len - 1):
                        self.optimizer_g.zero_grad()
                        pred = self.model(x_blurred_B1NN)
                        loss_g, loss_g1v, loss_g2v = self.criterion_g(pred, x_truth_B1NN, self.model_d)
                        loss_g.backward()
                        self.optimizer_g.step()
                        avg_loss_g += loss_g.item()
                        num_items_g += 1

                avg_loss_d = avg_loss_d / num_items_d
                avg_loss_g = avg_loss_g / num_items_g

                loss_record_d = np.append(loss_record_d, avg_loss_d.cpu().detach().numpy())
                loss_record_g = np.append(loss_record_g, avg_loss_g.cpu().detach().numpy())
                pbar.set_postfix({
                    "Loss_d": f"{avg_loss_d:.4f}",
                    "Loss_g": f"{avg_loss_g:.4f}"
                })
                pbar.update(1)

                if (epoch_num + 1) % configs.save_every == 0 or epoch_num == 0:
                    torch.save(self.model.state_dict(),
                               './saved_model/' + save_name + '_g_' + str(epoch_num + 1) + '.pth')
                    torch.save(self.model_d.state_dict(),
                               './saved_model/' + save_name + '_d_' + str(epoch_num + 1) + '.pth')

                    np.savez(
                        './saved_model/' + save_name + '_' + 'train_record_d.npz',
                        loss_record=loss_record_d)
                    np.savez('./saved_model/' + save_name + '_' + 'train_record_g.npz',loss_record=loss_record_g)
            return self.model, self.model_d
                    