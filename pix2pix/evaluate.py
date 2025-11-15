import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tqdm import tqdm
import numpy as np
import random

import config
from pix2pix.pair_image_dataset import PairImageDataset
from pix2pix.models.generator import Generator
from pix2pix.utils import save_checkpoint, load_checkpoint, generate_images


SEED = 423
torch.backends.cudnn.deterministic = True
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

BCE_LOSS = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()
L2_LOSS = nn.MSELoss()


def train_fn(discriminator, generator, loader, discriminator_optimizer, generator_optimizer,
             skip_generator_step_factor=0):
    collect_losses = {
        "D_real_bce_loss": [],
        "D_fake_bce_loss": [],
        "G_fake_bce_loss": [],
        "l1_loss": [],
        "l2_loss": [],
    }

    for idx, (x, y, angle) in enumerate(tqdm(loader, leave=True)):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        angle = angle.to(config.DEVICE)

        # Discriminator training.
        y_fake = generator(x, angle)
        G_bce_loss = 0
        if discriminator is not None:
            D_real = discriminator(x, y)
            D_fake = discriminator(x, y_fake.detach())
            D_real_bce_loss = BCE_LOSS(D_real - torch.mean(D_fake),
                                       torch.ones_like(D_real) * (1 - config.LABEL_SMOOTHING_D))
            D_fake_bce_loss = BCE_LOSS(D_fake - torch.mean(D_real), torch.ones_like(D_fake) * config.LABEL_SMOOTHING_D)
            D_loss = (D_real_bce_loss + D_fake_bce_loss) / 2

            discriminator_optimizer.zero_grad()
            D_loss.backward()
            discriminator_optimizer.step()

            collect_losses["D_real_bce_loss"].append(float(D_real_bce_loss))
            collect_losses["D_fake_bce_loss"].append(float(D_fake_bce_loss))

            if skip_generator_step_factor != 0 and (idx + 1) % skip_generator_step_factor != 0:
                continue

            D_real = discriminator(x, y).detach()
            D_fake = discriminator(x, y_fake)
            G_real_bce_loss = BCE_LOSS(D_real - torch.mean(D_fake), torch.ones_like(D_real) * config.LABEL_SMOOTHING_G)
            G_fake_bce_loss = BCE_LOSS(D_fake - torch.mean(D_real),
                                       torch.ones_like(D_fake) * (1 - config.LABEL_SMOOTHING_G))

            # fake
            G_bce_loss = (G_fake_bce_loss + G_real_bce_loss) * 0.5 * config.GAN_LAMBDA
        else:
            collect_losses["D_real_bce_loss"].append(float(0))
            collect_losses["D_fake_bce_loss"].append(float(0))

        # Generator training
        l1_loss = L1_LOSS(y_fake, y) * config.L1_LAMBDA
        l2_loss = torch.sqrt(L2_LOSS(y_fake, y)) * config.L2_LAMBDA
        G_loss = G_bce_loss + l1_loss + l2_loss
        generator_optimizer.zero_grad()
        G_loss.backward()
        generator_optimizer.step()

        collect_losses["G_fake_bce_loss"].append(float(G_bce_loss))
        collect_losses["l1_loss"].append(float(l1_loss))
        collect_losses["l2_loss"].append(float(l2_loss))

    print(f'losses: {np.mean(collect_losses["D_real_bce_loss"])} '
          f'{np.mean(collect_losses["D_fake_bce_loss"])} '
          f'{np.mean(collect_losses["G_fake_bce_loss"])} '
          f'{np.mean(collect_losses["l1_loss"])} '
          f'{np.mean(collect_losses["l2_loss"])}')


def evaluate():
    generator = Generator(1, dropout_encoder=config.DROPOUT_G_ENCODER, dropout_decoder=config.DROPOUT_G_DECODER)
    load_checkpoint(config.EVALUATE_CHECKPOINT_GEN, generator)

    generator.to(config.DEVICE)
    generator.eval()

    val_dataset = PairImageDataset(root_dir=config.VAL_DIR, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    folder = '../../eval'
    file_src = val_dataset.list_files_src
    with torch.no_grad():
        for idx, (x, y, angle, rg_angles, is_rg_exists) in enumerate(tqdm(val_loader, leave=True)):
            x, y, angle = x.to(config.DEVICE), y.to(config.DEVICE), angle.to(config.DEVICE)
            rg_angles, is_rg_exists = rg_angles.to(config.DEVICE), is_rg_exists.to(config.DEVICE)
            y_fake = generator(x, angle, rg_angles, is_rg_exists)
            y_fake = (y_fake + 1)  # to 0..1 range
            save_image(y_fake, os.path.join(folder, f"{file_src[idx].split('/')[-1]}"))


if __name__ == "__main__":
    evaluate()
