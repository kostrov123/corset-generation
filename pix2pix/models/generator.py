import torch
import torch.nn as nn
from torch.nn import ModuleList

from pix2pix import config


class Pix2PixBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: bool = True, act: str = "relu",
                 dropout_rate: float = 0.2, use_bn: bool = True):
        super(Pix2PixBlock, self).__init__()
        conv2d = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="replicate") \
            if down \
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

        if use_bn:
            self.conv = nn.Sequential(
                conv2d,
                nn.BatchNorm2d(out_channels),
                nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            )
        else:
            self.conv = nn.Sequential(
                conv2d,
                nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            )

        self.use_dropout = dropout_rate != 0
        self.dropout = nn.Dropout(dropout_rate)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, kernel_size=4, stride=2, dropout_encoder=0.2, dropout_decoder=0.5, w_dim = 128):
        super().__init__()
        self.encoder_start = 3
        self.encoder_end = 6
        self.decoder_start = 1
        self.decoder_end = 4

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size, stride, 1, padding_mode="replicate"),
            nn.LeakyReLU(0.2),
        )
        down_layers = []
        down_in_multipliers = [1, 2, 4, 8, 8, 8]
        down_out_multipliers = [2, 4, 8, 8, 8, 8]
        # self.angle_embeddings = nn.ModuleList([nn.Linear(2, features * i) for i in [1] + down_out_multipliers[:1]])
        self.z_transform = nn.Sequential(
            nn.Linear(36, w_dim),
            nn.Dropout(dropout_encoder),
            nn.LeakyReLU(0.2),
            nn.Linear(w_dim, w_dim),
            nn.Dropout(dropout_encoder),
            nn.LeakyReLU(0.2),
        )

        for i in range(6):
            down_layers.append(Pix2PixBlock(
                features * down_in_multipliers[i],
                features * down_out_multipliers[i],
                down=True,
                act="leaky",
                dropout_rate=dropout_encoder
            ))

        self.down_layers = ModuleList(down_layers)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="replicate"),
            nn.ReLU()
        )

        up_layers = []
        up_in_multipliers = [8, 16, 16, 16, 16, 8, 4]
        up_out_multipliers = [8, 8, 8, 8, 4, 2, 1]
        for i in range(7):
            up_layers.append(Pix2PixBlock(
                features * up_in_multipliers[i],
                features * up_out_multipliers[i],
                down=False, act="relu",
                dropout_rate=dropout_decoder
            ))

        self.up_layers = ModuleList(up_layers)

        self.rg_angles_encoder_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(w_dim, features * i * 2),
                nn.Dropout(dropout_encoder)
            ) for i in down_in_multipliers[self.encoder_start:self.encoder_end]])

        self.rg_angles_decoder_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(w_dim, features * i * 2),
                nn.Dropout(dropout_encoder)
            ) for i in (up_in_multipliers[self.decoder_start:self.decoder_end])])

        self.sigma_e = torch.nn.ParameterList([nn.Parameter(torch.ones((1, features * i, 1, 1), requires_grad=True).to(config.DEVICE))
                                               for i in down_in_multipliers[self.encoder_start:self.encoder_end]])
        self.sigma_d = torch.nn.ParameterList([nn.Parameter(torch.ones((1, features * i, 1, 1), requires_grad=True).to(config.DEVICE))
                                               for i in up_in_multipliers[self.decoder_start:self.decoder_end]])

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, angle: torch.Tensor, rg_angles: torch.Tensor,
                is_rg_exists: torch.Tensor) -> torch.Tensor:
        first_cnn_output = self.initial_down(x)
        if self.training:
            z = torch.randn_like(rg_angles)/100
        else:
            z = torch.zeros_like(rg_angles)

        z = torch.cat((rg_angles, z), dim=1)
        w = self.z_transform(z)
        down_x = [first_cnn_output]
        for id, layer in enumerate(self.down_layers):
            # Update cnn input
            if id >= self.encoder_start and id < self.encoder_end:
                cnn_in = down_x[-1]
                down_x[-1] = self.rg_angle_activation(cnn_in, w, id - self.decoder_start, is_rg_exists, is_encoder=True)
            cnn_output = layer(down_x[-1])
            down_x.append(cnn_output)

        bottleneck = self.bottleneck(down_x[-1])

        cnn_output = self.up_layers[0](bottleneck)
        up_x = [cnn_output]
        down_x = list(reversed(down_x))
        for i, layer in enumerate(self.up_layers[1:]):
            cnn_in = torch.cat([up_x[-1], down_x[i]], 1)
            if i >= self.decoder_start - 1 and i < self.decoder_end - 1:
                cnn_in = self.rg_angle_activation(cnn_in, w, i - self.encoder_start + 1, is_rg_exists, is_encoder=False)
            cnn_output = layer(cnn_in)
            up_x.append(cnn_output)

        cnn_in = torch.cat([up_x[-1], down_x[-1]], 1)
        # cnn_in = self.rg_angle_activation(cnn_in, w, len(self.up_layers) - 1, is_rg_exists, is_encoder=False)
        return self.final_up(cnn_in)

    def angle_activation(self, x: torch.Tensor, angle: torch.Tensor, layer: int) -> torch.Tensor:
        angle_embedding = self.angle_embeddings[layer](angle)
        bs, c, h, w = x.shape
        reshaped = torch.reshape(x, (bs * c, h, w))
        reshaped = torch.swapaxes(reshaped, 0, 2)
        rescaled_input = torch.mul(reshaped, torch.reshape(angle_embedding, (bs * c,)))
        rescaled_input = torch.swapaxes(rescaled_input, 0, 2)
        alfa = self.angle_multiplier[layer]
        x = (1 - alfa) * x + alfa * torch.reshape(rescaled_input, (bs, c, h, w))
        return x

    def rg_angle_activation(self, x: torch.Tensor, rg_angles: torch.Tensor, layer: int, is_rg_exists: torch.Tensor,
                            is_encoder: bool) -> torch.Tensor:
        if is_encoder:
            if len(self.sigma_e) <= layer:
                return x
            style_embedding = self.rg_angles_encoder_embeddings[layer](rg_angles)
            sigma = self.sigma_e[layer]
        else:
            if len(self.sigma_d) <= layer:
                return x
            style_embedding = self.rg_angles_decoder_embeddings[layer](rg_angles)
            sigma = self.sigma_d[layer]

        style_embedding_sig, style_embedding_bias = torch.split(style_embedding, style_embedding.shape[1]//2, dim=1)
        style_embedding_sig = style_embedding_sig + 1
        bs, c, h, w = x.shape
        noise = torch.randn((1,1,h,w),device=config.DEVICE)/100

        # if self.training:
        #    z = torch.randn_like(rg_angles)/100
        # else:
        #    z = torch.zeros_like(rg_angles)/100
        x = x + noise * torch.repeat_interleave(sigma, bs, dim=0)

        x_normalized = torch.layer_norm(x, [h, w])

        reshaped = torch.reshape(x_normalized, (bs * c, h, w))
        reshaped = torch.swapaxes(reshaped, 0, 2)
        rescaled_input = torch.mul(reshaped, torch.reshape(style_embedding_sig, (bs * c,))) + \
                         torch.reshape(style_embedding_bias, (bs * c,))
        rescaled_input = torch.swapaxes(rescaled_input, 0, 2)

        alfa = is_rg_exists.unsqueeze(-1).unsqueeze(-1)
        x = (1 - alfa) * x + alfa * torch.reshape(rescaled_input, (bs, c, h, w))
        # x = torch.reshape(rescaled_input, (bs, c, h, w))

        return x