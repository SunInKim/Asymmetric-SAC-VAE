import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from .base import BaseNetwork, create_linear_network, weights_init_xavier


class Gaussian(BaseNetwork):

    def __init__(self, input_dim, output_dim, hidden_units=[256, 256],
                 std=None, leaky_slope=0.2):
        super(Gaussian, self).__init__()
        self.net = create_linear_network(
            input_dim, 2*output_dim if std is None else output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(leaky_slope),
            initializer=weights_init_xavier)

        self.std = std

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)

        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return Normal(loc=mean, scale=std)


class ConstantGaussian(BaseNetwork):

    def __init__(self, output_dim, std=1.0):
        super(ConstantGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros((x.size(0), self.output_dim)).to(x)
        std = torch.ones((x.size(0), self.output_dim)).to(x) * self.std
        return Normal(loc=mean, scale=std)


class Decoder(BaseNetwork):

    def __init__(self, input_dim=256, output_dim=3, std=1.0, leaky_slope=0.2, bot_dim = 10):
        super(Decoder, self).__init__()
        self.std = std
        self.leaky_slope = leaky_slope


        self.convt1 = nn.ConvTranspose2d(input_dim, 512, 8).apply(weights_init_xavier)
        self.convt2 = nn.ConvTranspose2d(512, 512, 3, 2, 1).apply(weights_init_xavier)
        self.convt3 = nn.ConvTranspose2d(512, 512, 3, 2, 1, 1).apply(weights_init_xavier)
        self.convt4 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1).apply(weights_init_xavier)
        self.convt5 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1).apply(weights_init_xavier)
        self.convt6 = nn.ConvTranspose2d(128, 128, 5, 2, 2, 1).apply(weights_init_xavier)

        self.con1 = nn.Conv2d(128, 256, 3, 1, 1).apply(weights_init_xavier)
        self.con2 = nn.Conv2d(256, 256, 3, 1, 1).apply(weights_init_xavier)
        self.con3 = nn.Conv2d(256, 10, 3, 1, 1).apply(weights_init_xavier)

    def forward(self, x):
        
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x,  dim=-1)

        num_batches, latent_dim = x.size()

        x = x.view(num_batches, latent_dim, 1, 1)

        x = F.leaky_relu(self.convt1(x),negative_slope=self.leaky_slope)
        x = F.leaky_relu(self.convt2(x),negative_slope=self.leaky_slope)
        x = F.leaky_relu(self.convt3(x),negative_slope=self.leaky_slope)
        x = F.leaky_relu(self.convt4(x),negative_slope=self.leaky_slope)
        x = F.leaky_relu(self.convt5(x),negative_slope=self.leaky_slope)
        x = F.leaky_relu(self.convt6(x),negative_slope=self.leaky_slope)
        
        con = F.leaky_relu(self.con1(x),negative_slope=self.leaky_slope)
        con = F.leaky_relu(self.con2(con),negative_slope=self.leaky_slope)
        con = torch.tanh(self.con3(con))

        return Normal(loc=con, scale=torch.ones_like(con) * self.std)

class Encoder(BaseNetwork):

    def __init__(self, input_dim=10, output_dim=256, latent_dim=256, hidden_units=[256, 256], leaky_slope=0.2):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, 128, 5, 2, 2).apply(weights_init_xavier)
        self.conv2 = nn.Conv2d(128, 256, 3, 2, 1).apply(weights_init_xavier)
        self.conv3 = nn.Conv2d(256, 512, 3, 2, 1).apply(weights_init_xavier)
        self.conv4 = nn.Conv2d(512, 512, 3, 2, 1).apply(weights_init_xavier)
        self.conv5 = nn.Conv2d(512, 512, 3, 2, 1).apply(weights_init_xavier)
        self.conv6 = nn.Conv2d(512, output_dim, 8).apply(weights_init_xavier)

        self.leaky_slope = leaky_slope

        self.gau = Gaussian(
            output_dim, latent_dim, hidden_units, leaky_slope=leaky_slope)


    def forward(self, x):
        num_batches, C, H, W = x.size()

        x1 = F.leaky_relu(self.conv1(x),negative_slope=self.leaky_slope)
        x2 = F.leaky_relu(self.conv2(x1),negative_slope=self.leaky_slope)
        x3 = F.leaky_relu(self.conv3(x2),negative_slope=self.leaky_slope)
        x4 = F.leaky_relu(self.conv4(x3),negative_slope=self.leaky_slope)
        x5 = F.leaky_relu(self.conv5(x4),negative_slope=self.leaky_slope)
        x6 = F.leaky_relu(self.conv6(x5),negative_slope=self.leaky_slope)
        feature = x6.view(num_batches, -1)

        distribution = self.gau(feature)

        return feature, distribution

class LatentNetwork(BaseNetwork):

    def __init__(self, input_dim, action_shape, feature_dim=256,
                 latent_dim=256, hidden_units=[256, 256],
                 leaky_slope=0.2):
        super(LatentNetwork, self).__init__()
        self.encoder = Encoder(
            input_dim, feature_dim, latent_dim, hidden_units, leaky_slope=leaky_slope)
        self.decoder = Decoder(
            latent_dim, input_dim,
            std=np.sqrt(0.1), leaky_slope=leaky_slope)
