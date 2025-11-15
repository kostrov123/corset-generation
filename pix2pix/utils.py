import torch
from torchvision.utils import save_image

import config


def generate_images(generator, val_loader, epoch, folder):
    generator.eval()
    iterator = iter(val_loader)
    with torch.no_grad():
        for i in range(10):
            x, y, a, rg_angles, is_rg_exists = next(iterator)
            x, y, a = x.to(config.DEVICE), y.to(config.DEVICE), a.to(config.DEVICE)
            rg_angles, is_rg_exists = rg_angles.to(config.DEVICE), is_rg_exists.to(config.DEVICE)
            y_fake = generator(x, a, rg_angles, is_rg_exists )
            y_fake = (y_fake + 1) * 2  # to 0..1 range
            save_image(y_fake, folder + f"/generated_{epoch}_{i}.png")
            #save_image((x + 1) / 2, folder + f"/input_{epoch}.png")
            #save_image((y + 1) / 2, folder + f"/real_{epoch}.png")


def save_checkpoint(model, optimizer, filename="checkpoint.pt"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, filename)


"""
    Loads weights and biases only.
"""
def load_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr
