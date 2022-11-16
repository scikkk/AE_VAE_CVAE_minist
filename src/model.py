'''
Author: scikkk 203536673@qq.com
Date: 2022-11-09 16:32:40
LastEditors: scikkk
LastEditTime: 2022-11-14 13:35:56
Description: file content
'''
import torch
import torch.nn as nn
from utils import to_one_hot
"""
    This file contains main arch of models, including:
    - AutoEncoder
    - Variational AutoEncoder
    - Conditional Variational AutoEncoder
    
"""


class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size, decoder_type="AE", label_size=10) -> None:
        super(Encoder, self).__init__()
        if decoder_type == "AE":
            self.mu = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),)
        elif decoder_type == "VAE":
            self.mu = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),)
            self.sigma = nn.Sequential(nn.Linear(x_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),)
        elif decoder_type == "CVAE":
            self.mu = nn.Sequential(nn.Linear(x_dim+label_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),nn.ReLU(),)
            self.sigma = nn.Sequential(nn.Linear(x_dim+label_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size),)
        self.decoder_type = decoder_type;

    def forward(self, xs):
        # 实现编码器的forward过程 (5/100)
        mu = self.mu(xs)
        if self.decoder_type in ["VAE", "CVAE"]:
            sigma =  self.sigma(xs)
            return mu, sigma
        return mu


class Decoder(nn.Module):
    def __init__(self, x_dim, hidden_size, latent_size, decoder_type="AE") -> None:
        super(Decoder, self).__init__()
        if decoder_type == "AE":
            self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, x_dim),)
        elif decoder_type == "VAE":
            # 实现VAE的decoder (5/100)
            self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, x_dim),)  
        elif decoder_type == "CVAE":
            # 实现CVAE的decoder (5/100)
            self.decoder = nn.Sequential(nn.Linear(latent_size+10, hidden_size), nn.ReLU(), nn.Linear(hidden_size, x_dim),)  
        else:
            raise NotImplementedError
        self.decoder_type = decoder_type

    def forward(self, zs):
        # 实现decoder的decode部分, 注意对不同的decoder_type的处理 (10/100)
        if self.decoder_type in ["AE", "VAE"]:
            return self.decoder(zs)
        elif self.decoder_type == "CVAE":
            x = self.decoder(zs)
            return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        # 实现AE的forward过程(5/100)
        reco_x = self.decoder(z)
        return z, reco_x


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, xs):
        # 实现VAE的forward过程(10/100)
        mu, sigma = self.encoder(xs)
        z = self.reparameterize(mu, sigma)
        reco_x = self.decoder(z)
        return reco_x, z, mu, sigma

    def generate(self, latent_size=10):
        # sample from the latent space and concat label that you want to generate
        z = torch.randn(100, latent_size)
        reco_x = self.decoder(z) 
        return reco_x


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(ConditionalVariationalAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, xs, ys):
        # 实现 CVAE的forward过程(15/100)
        xs = torch.cat((xs, ys), dim = -1) # diff
        mu, sigma = self.encoder(xs)
        z = self.reparameterize(mu, sigma)
        z = torch.cat((z, ys), dim = -1) # diff
        reco_x = self.decoder(z) 
        return reco_x, z, mu, sigma
        
    def generate(self):
        # 从latent space随机采样并于期望的标记拼接
        z = torch.randn(100, 10)
        labels = torch.cat([torch.full(size=(10, 1), fill_value=label, dtype=torch.int64) for label in range(10)])
        y = to_one_hot(labels, num_class=10)
        z = torch.cat((z, y), dim = -1) # diff
        reco_x = self.decoder(z) 
        return reco_x
