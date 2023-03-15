"""Code adapted from https://github.com/arpitbansal297/Cold-Diffusion-Models"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.simplelayers import SkipConnection
import torch.nn.functional as F
import math
from monai.inferers import sliding_window_inference
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, binary_erosion, binary_dilation, binary_fill_holes


def extract(a, t, x_shape):
    """extract values of a at indices t, reshape to match x_shape"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=1,
        timesteps=1000,
        loss_type='l1',
        batch_size=4,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.batch_size = batch_size

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    def get_stepwise_denoise_fn_eval(self, time, direct_recon=False):
        # get a denoise function that uses the ddim sampling method to denoise at a specific timestep
        # useful for monai sliding window inference since it expects the function to return a single image the same
        # size as the input image

        def denoise_fn_eval(img):

            t = time
            batch_size = img.shape[0]

            while t:
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                x1_bar = self.denoise_fn((img, step))
                if direct_recon:
                    return x1_bar

                x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                x = img - xt_bar + xt_sub1_bar
                img = x
                t = t - 1

            return img

        return denoise_fn_eval

    @torch.no_grad()
    def validate(self, x1, x2, return_imgs=False):
        device = x1.device

        # limit the number of timesteps to 1/5 of the total timesteps to simulate a reasonable amount of input noise
        t = torch.randint(0, self.num_timesteps // 5, (1,), device=device).long()
        x_noisy = self.q_sample(x_start=x1, x_end=x2, t=t)

        # use sliding window inference to stride over the image and denoise it
        x_recon = sliding_window_inference(
            inputs=x_noisy,
            roi_size=self.image_size,
            sw_batch_size=self.batch_size,
            predictor=self.get_stepwise_denoise_fn_eval(t[0].item())
        )

        # get losses
        if self.loss_type == 'l1':
            loss = (x1 - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x1, x_recon)
        else:
            raise NotImplementedError()

        # return losses and images if requested
        if return_imgs:
            return loss, x_noisy, x_recon
        return loss

    def estimate_timestep_and_filter(self, noisy_img):

        noisy_img = noisy_img.detach().cpu().numpy()

        # smooth image
        smoothed_img = gaussian_filter(noisy_img, sigma=1, radius=5)

        # med filter image
        med_filt_img = median_filter(noisy_img, size=5)

        # get foreground using smoothed image
        mask = smoothed_img > np.percentile(smoothed_img, 1)
        mask = binary_erosion(mask, iterations=5)
        mask = binary_dilation(mask, iterations=5)
        mask = binary_fill_holes(mask)

        # get background (noise) statistics
        noise = noisy_img[~mask].std()

        # noise std is approximately equal to sqrt(1-alpha), get the closest timestep to this noise
        closest_t = np.abs(self.sqrt_one_minus_alphas_cumprod.cpu().numpy() - noise).argmin()

        return closest_t, mask, torch.from_numpy(smoothed_img).cuda(), torch.from_numpy(med_filt_img).cuda()

    @torch.no_grad()
    def test(self, x1, x2, timestep, return_imgs=False):
        device = x1.device

        t = torch.full((1,), timestep, dtype=torch.long, device=device)
        x_noisy = self.q_sample(x_start=x1, x_end=x2, t=t)

        # do not assume prior knowledge of timestep! Must estimate from noisy image
        t_pred, foreground_mask, gauss_filt_img, med_filt_img = self.estimate_timestep_and_filter(x_noisy)

        # use sliding window inference to stride over the image and denoise it
        x_recon = sliding_window_inference(
            inputs=x_noisy,
            roi_size=self.image_size,
            sw_batch_size=self.batch_size,
            predictor=self.get_stepwise_denoise_fn_eval(t_pred)
        )

        # also get the direct reconstruction without using ddim sampling method
        x_recon_direct = sliding_window_inference(
            inputs=x_noisy,
            roi_size=self.image_size,
            sw_batch_size=self.batch_size,
            predictor=self.get_stepwise_denoise_fn_eval(t_pred, direct_recon=True)
        )

        # get losses for all recon methods
        return_dict = {}
        image_dict = {}
        for recon_image, recon_name in zip(
                [x_noisy, x_recon, x_recon_direct, gauss_filt_img, med_filt_img],
                ['noisy_img', 'ddim_recon', 'direct_recon', 'gauss_filt', 'med_filt']
        ):
            # get l1 loss
            l1_loss = (x1 - recon_image).abs().mean()

            # also get PSNR
            mse = ((x1 - x_recon) ** 2).mean()
            if mse == 0:  # MSE is zero means no noise is present in the signal .
                psnr = 100
            else:
                psnr = 20 * torch.log10(x1.max() / torch.sqrt(mse))

            return_dict[recon_name] = {
                'l1_loss': l1_loss.item(),
                'PSNR': psnr.item()
            }
            if return_imgs:
                image_dict[recon_name] = recon_image.cpu().numpy()

        return_dict.update({
            't_pred': t_pred,
            't_actual': timestep,
        })
        if return_imgs:
            image_dict.update({
                'foreground_mask': foreground_mask,
                'orig_img': x1.cpu().numpy(),
            })

        return return_dict, image_dict

    def q_sample(self, x_start, x_end, t):
        # simply use the alphas to interpolate
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def p_losses(self, x_start, x_end, t):

        x_mix = self.q_sample(x_start=x_start, x_end=x_end, t=t)
        x_recon = self.denoise_fn((x_mix, t))
        if self.loss_type == 'l1':
            loss = (x_start - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x1, x2):
        b, c, d, h, w = x1.shape
        device = x1.device
        assert h == self.image_size and w == self.image_size and d == self.image_size,\
            f'height, width and depth of image must be {self.image_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x1, x2, t)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbSkipConnection(SkipConnection):
    """Adjusted skip connection to pass through time embeddings"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):

        y, time_emb = self.submodule(x)

        if self.mode == "cat":
            return torch.cat([x[0], y], dim=self.dim), time_emb
        if self.mode == "add":
            return torch.add(x[0], y), time_emb
        if self.mode == "mul":
            return torch.mul(x[0], y), time_emb
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


class TimeEmbConv(Convolution):
    """Adjusted convolution to pass through time embeddings"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input):
        x, time_emb = input
        x = super().forward(x)
        return x, time_emb


class TimeEmbResUnit(ResidualUnit):
    """Adjusted residual unit to add time embeddings"""

    def __init__(self, time_emb_dim, **kwargs):
        super().__init__(**kwargs)
        # only add time embeddings for intermediate layers
        if self.in_channels != 1 and self.out_channels != 1:
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, self.in_channels)
            )

    def forward(self, input):
        x, time_emb = input
        res = self.residual(x)

        # get time embedding and add to input if not first or last layer
        if self.in_channels != 1 and self.out_channels != 1:
            condition = self.time_mlp(time_emb)
            x = x + condition.view(condition.shape[0], condition.shape[1], 1, 1, 1)

        cx = self.conv(x)
        return cx + res, time_emb


class TimeEmbUNet3D(UNet):

    def __init__(
            self,
            spatial_dims,
            in_channels,
            out_channels,
            channels,
            strides,
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            dropout=0.0,
            bias=True,
            adn_ordering="NDA",
            dimensions=None
    ):

        assert num_res_units > 0, "num_res_units must be greater than 0 for this model."

        # time dimension is the first channel size
        self.time_dim = channels[0]

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
            dimensions=dimensions
        )

        # add time embedding model
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_dim * 4, self.time_dim)
        )

    def _get_connection_block(self, down_path, up_path, subblock):
        """Modified to use time embeddings"""

        return nn.Sequential(down_path, TimeEmbSkipConnection(submodule=subblock), up_path)

    def _get_down_layer(self, in_channels, out_channels, strides, is_top):
        """Modified to use time embeddings"""

        if self.num_res_units > 0:

            mod = TimeEmbResUnit(
                self.time_dim,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = TimeEmbConv(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_up_layer(self, in_channels, out_channels, strides, is_top):
        """Modified to use time embeddings"""

        conv = TimeEmbConv(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = TimeEmbResUnit(
                self.time_dim,
                spatial_dims=self.dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, input):
        x, time = input

        # get time embedding
        time_emb = self.time_mlp(time)

        # pass through model, convolutions will pass time embedding through, residual units will add to conv input
        x, time_emb = self.model((x, time_emb))

        # just return denoised output
        return x


# example usage
if __name__ == "__main__":

    model = TimeEmbUNet3D(
        in_channels=1,
        out_channels=1,
        spatial_dims=3,
        num_res_units=2,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2)
    )
    print(model)
    x = torch.randn(4, 1, 96, 96, 96)
    print(model((x, torch.LongTensor([500, 500, 500, 500]))).shape)
