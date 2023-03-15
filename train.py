"""Code adapted from https://github.com/arpitbansal297/Cold-Diffusion-Models"""
import argparse
import copy
import os
import torch
import json
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    RandFlipd,
    ScaleIntensityRangePercentilesd,
    RandRotate90d,
    ToTensord,
    RandSpatialCropSamplesd,
)
from monai.data import (
    DataLoader, Dataset
)

from model import GaussianDiffusion, TimeEmbUNet3D


def cycle_dl(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def save_images(noisy_img, recon_img, train_step, val_idx, base_path):
    # save cross-sectional images to base path with step
    _, _, d, h, w = noisy_img.shape

    f, axs = plt.subplots(2, 3, figsize=(8, 5))

    # visualize d axis (hw plane)
    axs[0, 0].imshow(noisy_img[0, 0, d // 2, :, :])
    axs[0, 0].set_title('Noisy Image (2-3 plane)')
    axs[1, 0].imshow(recon_img[0, 0, d // 2, :, :])
    axs[1, 0].set_title('Reconstructed Image (2-3 plane)')

    # visualize h axis (dw plane)
    axs[0, 1].imshow(noisy_img[0, 0, :, h // 2, :])
    axs[0, 1].set_title('Noisy Image (1-3 plane)')
    axs[1, 1].imshow(recon_img[0, 0, :, h // 2, :])
    axs[1, 1].set_title('Reconstructed Image (1-3 plane)')

    # visualize w axis (dh plane)
    axs[0, 2].imshow(noisy_img[0, 0, :, :, w // 2])
    axs[0, 2].set_title('Noisy Image (1-2 plane)')
    axs[1, 2].imshow(recon_img[0, 0, :, :, w // 2])
    axs[1, 2].set_title('Reconstructed Image (1-2 plane)')

    plt.savefig(os.path.join(base_path, f'img_{val_idx}_train_step_{train_step}.png'))


def train(args):
    # get json datalists
    with open(args.train_json, 'r') as f:
        train_data = json.load(f)
    with open(args.val_json, 'r') as f:
        val_data = json.load(f)
    
    # remove train item with only 28px in an axis...
    train_data = [dat for dat in train_data if 'IXI014-HH-1236-T2.nii.gz' not in dat['image']]

    # make output dir
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    print(f'train data length: {len(train_data)}')
    print(f'val data length: {len(val_data)}')

    # define train and val transforms
    train_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        ScaleIntensityRangePercentilesd(keys=['image'], lower=0.05, upper=99.95, b_min=-1, b_max=1),
        RandSpatialCropSamplesd(keys=['image'], roi_size=(96, 96, 96), num_samples=4, random_size=False),
        RandFlipd(keys=['image'], spatial_axis=[0], prob=0.1),
        RandFlipd(keys=['image'], spatial_axis=[1], prob=0.1),
        RandFlipd(keys=['image'], spatial_axis=[2], prob=0.1),
        RandRotate90d(keys=['image'], prob=0.1, spatial_axes=(0, 1)),
        ToTensord(keys=['image'])
    ])
    val_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        ScaleIntensityRangePercentilesd(keys=['image'], lower=0.05, upper=99.95, b_min=-1, b_max=1),
        ToTensord(keys=['image'])
    ])

    # datasets and dataloaders
    train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_ds = Dataset(data=val_data, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    # cycle train dataloader
    train_loader = cycle_dl(train_loader)

    # model params mimic those in original code
    model = TimeEmbUNet3D(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512),
        num_res_units=2,
        strides=(2, 2, 2)
    ).cuda()

    diffusion_model = GaussianDiffusion(
        model,
        image_size=96,
        channels=1,
        timesteps=args.timesteps,
        loss_type='l1',
        batch_size=args.batch_size
    ).cuda()

    # TODO: dataparallel here if needed

    # ema model
    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(diffusion_model)
    ema_model.load_state_dict(diffusion_model.state_dict())
    ema_model.eval()

    # params for ema and training
    update_ema_every = 10
    grad_accum_every = 2
    val_every = 2000
    step_start_ema = 2000

    # optimizer
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=args.lr)

    # training loop
    accum_loss = 0
    step = 0
    best_val_loss = 1000000
    best_val_step = 0
    while step < args.train_steps:
        denoise_loss = 0

        # gradient accumulation
        for i in range(grad_accum_every):
            data_1 = next(train_loader)['image']
            data_2 = torch.randn_like(data_1)

            data_1, data_2 = data_1.cuda(), data_2.cuda()
            loss = diffusion_model(data_1, data_2)
            if step % 100 == 0:
                print(f'step: {step}, loss: {loss.item()}')
            denoise_loss += loss.item()
            loss = loss / grad_accum_every
            loss.backward()

        accum_loss += denoise_loss / grad_accum_every
        optimizer.step()
        optimizer.zero_grad()

        # update ema model
        if step % update_ema_every == 0:
            if step < step_start_ema:
                ema_model.load_state_dict(diffusion_model.state_dict())
            else:
                ema.update_model_average(ema_model, diffusion_model)

        # validation
        if step % val_every == 0 and step != 0:
            print(f'validation at step: {step}')
            ema_model.eval()
            val_denoise_loss = 0
            for val_idx, val_data in enumerate(val_loader):
                data_1 = val_data['image']
                data_2 = torch.randn_like(data_1)

                data_1, data_2 = data_1.cuda(), data_2.cuda()
                # visualize reconstructions for the first 3 images in val
                if val_idx < 3:
                    loss, noisy_img, recon_img = ema_model.validate(data_1, data_2, return_imgs=True)
                    noisy_img = noisy_img.detach().cpu().numpy()
                    recon_img = recon_img.detach().cpu().numpy()
                    save_images(noisy_img, recon_img, step, val_idx, args.log_path)
                else:
                    loss = ema_model.validate(data_1, data_2)
                val_denoise_loss += loss.item()

            val_denoise_loss = val_denoise_loss / len(val_loader)
            print(f'validation loss: {val_denoise_loss}')
            print(f'mean train loss: {accum_loss / val_every}')
            accum_loss = 0

            # save best model based on validation reconstruction loss
            if val_denoise_loss < best_val_loss:
                best_val_loss = val_denoise_loss
                best_val_step = step
                torch.save(ema_model.state_dict(), os.path.join(args.log_path, 'best_model.pth'))

        # save model 10 times during training
        if (step + 1) % (args.train_steps // 10) == 0:
            torch.save(ema_model.state_dict(), os.path.join(args.log_path, f'model_step_{step}.pth'))

        step += 1

    print(f'best validation loss: {best_val_loss} at step: {best_val_step}')
    print('done training!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str)
    parser.add_argument('--val_json', type=str)
    parser.add_argument('--test_json', type=str)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--ema_decay', default=0.995, type=float)
    parser.add_argument('--train_steps', default=700000, type=int)
    parser.add_argument('--timesteps', default=100, type=int)
    parser.add_argument('--lr', default=0.00002, type=float)

    args = parser.parse_args()

    print('===== arguments for this run =====')
    print(vars(args))

    train(args)
