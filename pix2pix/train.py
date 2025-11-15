import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger

from tqdm import tqdm
import numpy as np
import random

import config
from pix2pix.pair_image_dataset import PairImageDataset
from pix2pix.models.discriminator import Discriminator
from pix2pix.models.generator import Generator
from pix2pix.utils import save_checkpoint, load_checkpoint, generate_images

# Фиксируем сиды для воспроизводимости экспериментов
SEED = 423
torch.backends.cudnn.deterministic = True
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Метрики
BCE_LOSS = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()
L2_LOSS = nn.MSELoss()


def train_fn(discriminator, generator, loader, discriminator_optimizer, generator_optimizer,
             skip_generator_step_factor=0):
    """
    Функция, выполняемая в цикле обучения.
    skip_generator_step_factor - сколько шагов дискриминатор выполнить на 1 шаг генератора (0 - 1 к 1)
    """
    collect_losses = {
        "D_real_bce_loss": [],
        "D_fake_bce_loss": [],
        "G_fake_bce_loss": [],
        "l1_loss": [],
        "l2_loss": [],
    }

    #  (x, y, angle, rg_angles, is_rg_exists) с выхода PairImageDataset
    for idx, (x, y, angle, rg_angles, is_rg_exists) in enumerate(tqdm(loader, leave=True)):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        angle = angle.to(config.DEVICE)
        rg_angles = rg_angles.to(config.DEVICE)
        is_rg_exists = is_rg_exists.to(config.DEVICE)

        # Discriminator training.
        y_fake = generator(x, angle, rg_angles, is_rg_exists)
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


def train():
    logger = CSVLogger("logs", name="corsets", flush_logs_every_n_steps=1)
    logger.log_hyperparams({
        'LR': config.LR,
        'LR_PATIENCE': config.LR_PATIENCE,
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'GAN_LAMBDA': config.GAN_LAMBDA,
        'L1_LAMBDA': config.L1_LAMBDA,
        'L2_LAMBDA': config.L2_LAMBDA,
        'DROPOUT_G_ENCODER': config.DROPOUT_G_ENCODER,
        'DROPOUT_G_DECODER': config.DROPOUT_G_DECODER,
    })
    generator = Generator(1, dropout_encoder=config.DROPOUT_G_ENCODER, dropout_decoder=config.DROPOUT_G_DECODER)
    if config.LOAD_GENERATOR_MODEL:
        load_checkpoint(
            config.PRETRAIN_CHECKPOINT_GEN, generator
        )

    generator.to(config.DEVICE)

    if not config.TRAIN_GENERATOR_ONLY:
        discriminator = Discriminator()
        discriminator.to(config.DEVICE)
        discriminator.train()
        # betas are fully depend on Batch size. For small batches -> small beta
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.LR, betas=(0.99, 0.999), )
        discriminator_scheduler = ReduceLROnPlateau(discriminator_optimizer, 'min', factor=0.5,
                                                    patience=config.LR_PATIENCE, cooldown=2, min_lr=1e-7)
    else:
        discriminator = None
        discriminator_optimizer = None
        discriminator_scheduler = None

    generator_optimizer = optim.Adam(generator.parameters(), lr=config.LR, betas=(0.99, 0.999))
    generator_scheduler = ReduceLROnPlateau(generator_optimizer, 'min', factor=0.7 if config.TRAIN_GENERATOR_ONLY else 0.5, patience=config.LR_PATIENCE,
                                            cooldown=2, min_lr=1e-7, verbose=True)

    # train_concat_dataset = PairImageDataset(root_dir=config.TRAIN_OLD_VERSION_DIR, is_train=True)
    train_dataset = PairImageDataset(root_dir=config.TRAIN_DIR, is_train=True)
    # train_old_dataset = PairImageDataset(root_dir=config.TRAIN_OLD_VERSION_DIR, is_train=True)
    # train_concat_dataset = torch.utils.data.ConcatDataset([train_dataset, train_old_dataset])
    print(f'train dataset size: {len(train_dataset)}')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    val_dataset = PairImageDataset(root_dir=config.VAL_DIR, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        generator.train()
        train_fn(
            discriminator, generator, train_loader, discriminator_optimizer, generator_optimizer
        )

        if config.SAVE_MODEL and epoch % config.CHECKPOINT_SAVE_STEP == 0:
            ckpt_dir = os.path.join(config.CHECKPOINT_DIR, f'{logger.version}')
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            gen_filepath = os.path.join(ckpt_dir, f'generator_{epoch}.pt')
            save_checkpoint(generator, generator_optimizer, filename=gen_filepath)
            if not config.TRAIN_GENERATOR_ONLY:
                disc_filepath = os.path.join(ckpt_dir, f'disc_{epoch}.pt')
                save_checkpoint(discriminator, discriminator_optimizer, filename=disc_filepath)

        generator.eval()
        if epoch % 10 == 0:
            generate_images(generator, val_loader, epoch, folder="evaluation")
        losses_mae = []
        losses_mse = []

        with torch.no_grad():
            for idx, (x, y, angle, rg_angles, is_rg_exists ) in enumerate(tqdm(val_loader, leave=True)):
                x, y, angle = x.to(config.DEVICE), y.to(config.DEVICE), angle.to(config.DEVICE)
                rg_angles, is_rg_exists = rg_angles.to(config.DEVICE), is_rg_exists.to(config.DEVICE)
                y_fake = generator(x, angle, rg_angles, is_rg_exists)
                losses_mae.append(float(torch.abs((y_fake - y)).mean()))
                losses_mse.append(float(torch.pow((y_fake - y), 2).mean()))

        mae = np.mean(losses_mae)
        mse = np.mean(losses_mse)
        print(f'epoch {epoch}: mae: {mae}; mse: {mse}')
        logger.log_metrics({'mae': mae, 'mse': mse}, epoch)
        generator_scheduler.step(mae)
        if not config.TRAIN_GENERATOR_ONLY:
            discriminator_scheduler.step(mae)


if __name__ == "__main__":
    train()
