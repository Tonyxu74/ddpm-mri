"""Code adapted from https://github.com/arpitbansal297/Cold-Diffusion-Models"""
import argparse
import os
import torch
import json
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)
from monai.data import (
    DataLoader, Dataset
)

from model import GaussianDiffusion, TimeEmbUNet3D


def make_summary_figure(image_dict, test_idx, base_path):
    # make summary figure for report
    d = image_dict['orig_img'].shape[-1]

    f, axs = plt.subplots(1, 6, figsize=(15, 3))

    for idx, key in enumerate(['orig_img', 'noisy_img', 'gauss_filt', 'med_filt', 'direct_recon', 'ddim_recon']):
        if key != 'noisy_img':
            axs[idx].imshow(image_dict[key][0, 0, :, :, d // 2], cmap='gray', vmin=-1, vmax=1)
            axs[idx].axis('off')
        else:
            axs[idx].imshow(image_dict[key][0, 0, :, :, d // 2], cmap='gray')
            axs[idx].axis('off')

    plt.savefig(os.path.join(base_path, f'img_{test_idx}_summary.png'), bbox_inches='tight')
    plt.close()


def vis_foreground_mask(image_dict, test_idx, base_path):
    # vis foreground mask creation for report
    d = image_dict['orig_img'].shape[-1]

    f, axs = plt.subplots(1, 3, figsize=(7, 3))

    for idx, key in enumerate(['orig_img', 'noisy_img', 'foreground_mask']):
        if key == 'orig_img':
            axs[idx].imshow(image_dict[key][0, 0, :, :, d // 2], cmap='gray', vmin=-1, vmax=1)
            axs[idx].axis('off')
        else:
            axs[idx].imshow(image_dict[key][0, 0, :, :, d // 2], cmap='gray')
            axs[idx].axis('off')

    plt.savefig(os.path.join(base_path, f'img_{test_idx}_fg_mask.png'), bbox_inches='tight')
    plt.close()


def test_and_vis(args):
    # get json datalists
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    # make output dir
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    print(f'test data length: {len(test_data)}')

    # define train and val transforms
    test_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        ScaleIntensityRangePercentilesd(keys=['image'], lower=0.05, upper=99.95, b_min=-1, b_max=1),
        ToTensord(keys=['image'])
    ])

    # datasets and dataloaders
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # model params mimic those in original code
    model = TimeEmbUNet3D(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512),
        num_res_units=2,
        strides=(2, 2, 2)
    ).cuda()

    ema_model = GaussianDiffusion(
        model,
        (96, 96, 96),
        channels=1,
        timesteps=args.timesteps,
        loss_type='l1',
        batch_size=args.batch_size
    ).cuda()

    ema_model.load_state_dict(torch.load(args.model_path))
    ema_model.eval()

    timestep = args.noise_timestep

    # testing loop
    recon_l1 = 0
    recon_psnr = 0
    direct_recon_l1 = 0
    direct_recon_psnr = 0
    gauss_l1 = 0
    gauss_psnr = 0
    med_l1 = 0
    med_psnr = 0

    for test_idx, test_data in enumerate(test_loader):
        print('======== on test idx: ', test_idx, ' ========', flush=True)
        data_1 = test_data['image']
        data_2 = torch.randn_like(data_1)

        data_1, data_2 = data_1.cuda(), data_2.cuda()
        # # visualize reconstructions for the first few images in test
        if test_idx < 10:
            return_dict, image_dict = ema_model.test(data_1, data_2, timestep, return_imgs=True)
            make_summary_figure(image_dict, test_idx, args.log_path)
            vis_foreground_mask(image_dict, test_idx, args.log_path)
        else:
            return_dict, image_dict = ema_model.test(data_1, data_2, timestep, return_imgs=False)

        recon_l1 += return_dict['ddim_recon']['l1_loss']
        recon_psnr += return_dict['ddim_recon']['PSNR']
        direct_recon_l1 += return_dict['direct_recon']['l1_loss']
        direct_recon_psnr += return_dict['direct_recon']['PSNR']
        gauss_l1 += return_dict['gauss_filt']['l1_loss']
        gauss_psnr += return_dict['gauss_filt']['PSNR']
        med_l1 += return_dict['med_filt']['l1_loss']
        med_psnr += return_dict['med_filt']['PSNR']

        print(return_dict, flush=True)

    for value, name in zip(
            [recon_l1, recon_psnr, direct_recon_l1, direct_recon_psnr, gauss_l1, gauss_psnr, med_l1, med_psnr],
            ['recon l1', 'recon PSNR', 'direct recon l1', 'direct recon PSNR',
             'gauss l1', 'gauss PSNR', 'med l1', 'med PSNR']
    ):
        print(f'test {name}: {value / len(test_loader)}')

    print('done testing!')


def training_progress(args):
    # grab a few epochs and make images of the training progress
    pass


def pred_vs_actual_timestep(args):
    # make this if performance is severely degraded
    pass


def generate_sample(args):
    # create a sample from pure noise
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_json', type=str)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--noise_timestep', default=20, type=int)
    parser.add_argument('--timesteps', default=100, type=int)

    args = parser.parse_args()

    print('===== arguments for this run =====')
    print(vars(args))

    test_and_vis(args)
