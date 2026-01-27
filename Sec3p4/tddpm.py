# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import configs
import tqdm
from torch.optim import Adam
from configs import alpha, bar_alpha, beta, sigma
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


class Downsample(nn.Module):
    """Strided conv downsample (faster than attention, stable)."""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv (fast, avoids checkerboard artifacts)."""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


def _gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    # Use <=8 groups; fall back safely if channels is small / not divisible.
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class CondResBlock(nn.Module):

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        y_ch: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # Fuse spatial conditioning (post-conv1) through 1x1 conv.
        self.fuse = nn.Conv2d(out_ch + y_ch, out_ch, kernel_size=1)

        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # FiLM: embedding -> per-channel scale, shift
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, y_feat: torch.Tensor) -> torch.Tensor:
        # emb: (B, emb_dim), y_feat: (B, y_ch, H, W) (or will be resized)
        h = self.conv1(F.silu(self.norm1(x)))

        if y_feat is not None:
            if y_feat.shape[-2:] != h.shape[-2:]:
                y_feat = F.interpolate(y_feat, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = self.fuse(torch.cat([h, y_feat], dim=1))

        scale_shift = self.emb_proj(emb)  # (B, 2*out_ch)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))

        return h + self.skip(x)


class denoiseNet(nn.Module):
    """

    Inputs:
      y   : (B, 5, 128, 32)  measurement
      x_t : (B, 1, 32, 32)   noisy state at timestep t
      t   : (B,)             timestep tensor (int or float)

    Output:
      pred: (B, 1, 32, 32)   typically epsilon (noise) or v
    """
    def __init__(
        self,
        embed_dim: int = 256,
        base_ch: int = 64,
        dropout: float = 0.0,  # set small like 0.05-0.1 if you see overfit; 0 is fastest
    ):
        super().__init__()

        self.info_enc = SetTransformerEncoder()
        
        # ---- time embedding structure (exactly as requested) ----
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # A small MLP after the required structure (recommended for robustness)
        self.temb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ---- y encoder: convert (128x32) -> (32x32), then build pyramid (32/16/8/4) ----
        self.y_stem = nn.Conv2d(5, base_ch, kernel_size=3, padding=1)

        # Reduce height 128->64->32, keep width 32 (stride (2,1))
        self.y_down_h1 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=(2, 1), padding=1)
        self.y_down_h2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=(2, 1), padding=1)

        # Pyramid
        self.y_down32_16 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)      # 32->16
        self.y_down16_8  = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)  # 16->8
        self.y_down8_4   = nn.Conv2d(base_ch * 4, base_ch * 8, kernel_size=3, stride=2, padding=1)  # 8->4

        self.m_stem = nn.Conv2d(1, base_ch, kernel_size=3, padding=1)  # -> (B, base, 32, 32)
        self.m_down32_16 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)  # -> (B, 2base, 16, 16)
        self.m_down16_8 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)  # -> (B, 4base, 8, 8)
        self.m_down8_4 = nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1)  # -> (B, 8base, 4, 4)

        # --- fuse y_pyramid with m_pyramid at each scale (concat then 1x1) ---
        self.fuse32 = nn.Conv2d(base_ch + base_ch, base_ch, kernel_size=1)
        self.fuse16 = nn.Conv2d(base_ch * 2 + base_ch * 2, base_ch * 2, kernel_size=1)
        self.fuse8 = nn.Conv2d(base_ch * 4 + base_ch * 4, base_ch * 4, kernel_size=1)
        self.fuse4 = nn.Conv2d(base_ch * 8 + base_ch * 8, base_ch * 8, kernel_size=1)

        # --- global embed also includes y_img bottleneck ---
        self.m_global = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(base_ch * 8, embed_dim, kernel_size=1),
        )
        
        # Global y embedding to blend with time embedding (robust conditioning)
        self.y_global = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(base_ch * 8, embed_dim, kernel_size=1),
        )

        # ---- x_t U-Net trunk ----
        self.x_in = nn.Conv2d(1, base_ch, kernel_size=3, padding=1)

        # Down: 32 -> 16 -> 8 -> 4
        self.down1a = CondResBlock(base_ch, base_ch, emb_dim=embed_dim, y_ch=base_ch, dropout=dropout)
        self.down1b = CondResBlock(base_ch, base_ch, emb_dim=embed_dim, y_ch=base_ch, dropout=dropout)
        self.down1_ds = Downsample(base_ch)

        self.down2a = CondResBlock(base_ch, base_ch * 2, emb_dim=embed_dim, y_ch=base_ch * 2, dropout=dropout)
        self.down2b = CondResBlock(base_ch * 2, base_ch * 2, emb_dim=embed_dim, y_ch=base_ch * 2, dropout=dropout)
        self.down2_ds = Downsample(base_ch * 2)

        self.down3a = CondResBlock(base_ch * 2, base_ch * 4, emb_dim=embed_dim, y_ch=base_ch * 4, dropout=dropout)
        self.down3b = CondResBlock(base_ch * 4, base_ch * 4, emb_dim=embed_dim, y_ch=base_ch * 4, dropout=dropout)
        self.down3_ds = Downsample(base_ch * 4)

        self.down4a = CondResBlock(base_ch * 4, base_ch * 8, emb_dim=embed_dim, y_ch=base_ch * 8, dropout=dropout)
        self.down4b = CondResBlock(base_ch * 8, base_ch * 8, emb_dim=embed_dim, y_ch=base_ch * 8, dropout=dropout)

        # Middle (no attention; still use two ResBlocks)
        self.mid1 = CondResBlock(base_ch * 8, base_ch * 8, emb_dim=embed_dim, y_ch=base_ch * 8, dropout=dropout)
        self.mid2 = CondResBlock(base_ch * 8, base_ch * 8, emb_dim=embed_dim, y_ch=base_ch * 8, dropout=dropout)

        # Up: 4 -> 8 -> 16 -> 32
        self.up3_us = Upsample(base_ch * 8)
        self.up3a = CondResBlock(base_ch * 8 + base_ch * 4, base_ch * 4, emb_dim=embed_dim, y_ch=base_ch * 4, dropout=dropout)
        self.up3b = CondResBlock(base_ch * 4, base_ch * 4, emb_dim=embed_dim, y_ch=base_ch * 4, dropout=dropout)

        self.up2_us = Upsample(base_ch * 4)
        self.up2a = CondResBlock(base_ch * 4 + base_ch * 2, base_ch * 2, emb_dim=embed_dim, y_ch=base_ch * 2, dropout=dropout)
        self.up2b = CondResBlock(base_ch * 2, base_ch * 2, emb_dim=embed_dim, y_ch=base_ch * 2, dropout=dropout)

        self.up1_us = Upsample(base_ch * 2)
        self.up1a = CondResBlock(base_ch * 2 + base_ch, base_ch, emb_dim=embed_dim, y_ch=base_ch, dropout=dropout)
        self.up1b = CondResBlock(base_ch, base_ch, emb_dim=embed_dim, y_ch=base_ch, dropout=dropout)

        self.out = nn.Sequential(
            _gn(base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 1, kernel_size=3, padding=1),
        )

        # Good default init for stability (optional but robust)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_y(self, y: torch.Tensor):
        # y: (B, 5, 128, 32)
        y0 = F.silu(self.y_stem(y))             # (B, base, 128, 32)
        y0 = F.silu(self.y_down_h1(y0))         # (B, base, 64, 32)
        y32 = F.silu(self.y_down_h2(y0))        # (B, base, 32, 32)

        y16 = F.silu(self.y_down32_16(y32))     # (B, 2base, 16, 16)
        y8  = F.silu(self.y_down16_8(y16))      # (B, 4base, 8, 8)
        y4  = F.silu(self.y_down8_4(y8))        # (B, 8base, 4, 4)

        yglob = self.y_global(y4).squeeze(-1).squeeze(-1)  # (B, embed_dim)
        return y32, y16, y8, y4, yglob

    def _encode_m(self, y_img: torch.Tensor):
        # y_img: (B,1,32,32)
        m32 = F.silu(self.m_stem(y_img))  # (B, base, 32, 32)
        m16 = F.silu(self.m_down32_16(m32))  # (B, 2base, 16, 16)
        m8 = F.silu(self.m_down16_8(m16))  # (B, 4base, 8, 8)
        m4 = F.silu(self.m_down8_4(m8))  # (B, 8base, 4, 4)
        mglob = self.m_global(m4).squeeze(-1).squeeze(-1)  # (B, embed_dim)
        return m32, m16, m8, m4, mglob


    def forward(self, y: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, ensemble_info = None) -> torch.Tensor:
        # Ensure float time input for Fourier features
        if t.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8):
            t = t.float()

        y32, y16, y8, y4, yglob = self._encode_y(y)

        # new measurement pyramid
        if ensemble_info is None:
            ensemble_info = self.info_enc(y).repeat(y.shape[0], 1, 1, 1)

        m32, m16, m8, m4, mglob = self._encode_m(ensemble_info)

        # fuse per-scale conditioning
        y32 = F.silu(self.fuse32(torch.cat([y32, m32], dim=1)))
        y16 = F.silu(self.fuse16(torch.cat([y16, m16], dim=1)))
        y8 = F.silu(self.fuse8(torch.cat([y8, m8], dim=1)))
        y4 = F.silu(self.fuse4(torch.cat([y4, m4], dim=1)))

        # embeddings
        temb = self.temb_mlp(self.embed(t))
        emb = temb + yglob + mglob

        h = self.x_in(x_t)

        # Down 32
        h = self.down1a(h, emb, y32)
        h = self.down1b(h, emb, y32)
        skip1 = h
        h = self.down1_ds(h)  # -> 16

        # Down 16
        h = self.down2a(h, emb, y16)
        h = self.down2b(h, emb, y16)
        skip2 = h
        h = self.down2_ds(h)  # -> 8

        # Down 8
        h = self.down3a(h, emb, y8)
        h = self.down3b(h, emb, y8)
        skip3 = h
        h = self.down3_ds(h)  # -> 4

        # Down 4
        h = self.down4a(h, emb, y4)
        h = self.down4b(h, emb, y4)

        # Mid
        h = self.mid1(h, emb, y4)
        h = self.mid2(h, emb, y4)

        # Up to 8
        h = self.up3_us(h)
        h = torch.cat([h, skip3], dim=1)
        h = self.up3a(h, emb, y8)
        h = self.up3b(h, emb, y8)

        # Up to 16
        h = self.up2_us(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.up2a(h, emb, y16)
        h = self.up2b(h, emb, y16)

        # Up to 32
        h = self.up1_us(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.up1a(h, emb, y32)
        h = self.up1b(h, emb, y32)

        return self.out(h)

class tddpm:
    def __init__(self, device=configs.device, epoch=configs.epoch, total_steps=configs.total_steps, lr=configs.lr,
                 batch_size=configs.batch_size):
        super().__init__()
        self.epoch = epoch
        self.device = device
        # self.denoise_model = torch.nn.DataParallel(denoiseNet()).to(self.device)
        # self.denoise_model_noparallel = denoiseNet().to(self.device)
        self.denoise_model = denoiseNet().to(self.device)

        self.total_steps = total_steps
        self.optimizer = Adam(self.denoise_model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.MSE = nn.MSELoss()

    def load_denoise_model_noparallel(self, ckpt):
        new_state_dict = {}
        for key, value in ckpt.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value
        self.denoise_model.load_state_dict(new_state_dict)


    def train(self, MNIST_dataset, model_type, save_name='test', init_ckpt = None, rep_len=8, exclude = None):

        loss_record = np.array([])
        if init_ckpt is not None:
            self.denoise_model.load_state_dict(init_ckpt)
        with tqdm.tqdm(total=self.epoch, desc=f"Epoch ", unit="batch") as pbar:
            for epoch_num in range(self.epoch):
                avg_loss = torch.tensor(0.0).to(self.device)
                num_items = 0
                for k in range(100 * rep_len):
                    # for (x_truth_B1NN,y1), (x_blurred_B1NN,y2) in zip(data_loader_truth_list[j],data_loader_blurred_list[j]):
                    x_truth_B1NN, x_blurred_B1NN = MNIST_dataset.get_mixed_data(self.batch_size, exclude=exclude)

                    x_truth_B1NN = x_truth_B1NN.to(self.device)
                    x_blurred_B1NN = x_blurred_B1NN.to(self.device)

                    if model_type == 'DDPM':
                        loss = self.loss_fn(self.denoise_model, x_truth_B1NN, x_blurred_B1NN)
                    elif model_type == 'FM':
                        loss = self.loss_fn_FM(self.denoise_model, x_truth_B1NN, x_blurred_B1NN)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    avg_loss += loss
                    num_items += 1
                avg_loss = avg_loss / num_items
                loss_record = np.append(loss_record, avg_loss.item())
                pbar.set_postfix({
                    "Loss": f"{avg_loss:.4f}"
                })
                pbar.update(1)

                if (epoch_num + 1) % configs.save_every == 0 or epoch_num == 0:
                    torch.save(self.denoise_model.state_dict(),
                               './saved_model/' + save_name + '_' + str(epoch_num + 1) + '.pth')
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

        xt_B1NN = (x_truth_B1NN * sqrt_bar_alpha_list[:, None, None, None]) + eps * sqrt_invbar_alpha_list[:, None,
                                                                                    None, None]

        # combinedx_B2NN = torch.cat([xt_B1NN, x_blurred_B1NN], dim=1)
        eps_est = model(x_blurred_B1NN, xt_B1NN, random_t)

        loss = self.MSE(eps_est, eps)

        return loss

    def loss_fn_FM(self, model, x_truth_B1NN, x_blurred_B1NN):
        random_t = torch.rand(x_truth_B1NN.shape[0]).to(self.device)
        x_init_B1NN = torch.randn_like(x_truth_B1NN)
        v = x_truth_B1NN - x_init_B1NN
        random_t_mat = random_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        xt_B1NN = random_t_mat * x_truth_B1NN + (1 - random_t_mat) * x_init_B1NN
        # combinedx_B2NN = torch.cat([xt_B1NN, x_blurred_B1NN], dim=1)
        u_est = model(x_blurred_B1NN, xt_B1NN, random_t)

        loss = self.MSE(u_est, v)
        return loss

    def sampler(self, model, x_blurred_B1NN):
        batch_size = x_blurred_B1NN.shape[0]
        x_prev = torch.randn(batch_size, 1, 32, 32, device=self.device)
        info_embed = model.info_enc(x_blurred_B1NN).repeat(x_blurred_B1NN.shape[0], 1, 1, 1)
        for i in range(self.total_steps):
            t = (self.total_steps - i)
            if t > 1:
                z = torch.randn(batch_size, 1, 32, 32, device=self.device)
            else:
                z = torch.zeros(batch_size, 1, 32, 32, device=self.device)
            t_tensor = torch.ones(batch_size).to(self.device) * t
            with torch.no_grad():
                # modelinp = torch.cat([x_prev, x_blurred_B1NN], dim=1)
                x = 1 / np.sqrt(alpha(t)) * (
                        x_prev - (1 - alpha(t)) / (np.sqrt(1 - bar_alpha(t))) *
                        model(x_blurred_B1NN,x_prev, t_tensor,ensemble_info=info_embed)) + sigma(
                    t) * z

            x_prev = x
        return x_prev

    def sampler_FM(self, model, x_blurred_B1NN, steps_num=None):
        if steps_num is None:
            steps_num = self.total_steps

        info_embed = model.info_enc(x_blurred_B1NN).repeat(x_blurred_B1NN.shape[0], 1, 1, 1)
        batch_size = x_blurred_B1NN.shape[0]
        x_prev = torch.randn(batch_size, 1, 32, 32, device=self.device)
        for i in range(steps_num):
            t = i / steps_num
            t_tensor = torch.ones(batch_size).to(self.device) * t
            with torch.no_grad():
                # modelinp = torch.cat([x_prev, x_blurred_B1NN], dim=1)
                v_est = model(x_blurred_B1NN,x_prev, t_tensor,ensemble_info=info_embed)
            x_prev = x_prev + v_est / steps_num

        return x_prev


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=True):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            # ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class SetTransformerEncoder(nn.Module):
    """
    Input:  (B, 5, 128, 32)
    Output: (B, 128)
    """
    def __init__(self, in_ch: int = 5, out_dim: int = 128, channels=(32, 64, 128, 256), gn_groups=8):
        super().__init__()

        c1, c2, c3, c4 = channels

        # Note: padding=1 keeps "same-ish" spatial size when stride=1,
        # and halves (approximately) when stride=2.
        self.conv1 = nn.Conv2d(in_ch, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=min(gn_groups, c1), num_channels=c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=min(gn_groups, c2), num_channels=c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn3   = nn.GroupNorm(num_groups=min(gn_groups, c3), num_channels=c3)

        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn4   = nn.GroupNorm(num_groups=min(gn_groups, c4), num_channels=c4)

        # Pool -> flatten -> linear head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(c4, out_dim)

        self.Settransformer = SetTransformer(128, 32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, 128, 32)
        x = F.silu(self.gn1(self.conv1(x)))  # (B, c1, 128, 32)
        x = F.silu(self.gn2(self.conv2(x)))  # (B, c2, 64, 16)
        x = F.silu(self.gn3(self.conv3(x)))  # (B, c3, 32, 8)
        x = F.silu(self.gn4(self.conv4(x)))  # (B, c4, 16, 4)

        x = self.pool(x)                     # (B, c4, 1, 1)
        x = torch.flatten(x, 1)              # (B, c4)
        x = self.fc(x)                       # (B, 128)

        x = self.Settransformer(x.unsqueeze(0)).unsqueeze(0)
        return x
