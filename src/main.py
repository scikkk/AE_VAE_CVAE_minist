'''
Author: scikkk 203536673@qq.com
Date: 2022-11-09 16:48:32
LastEditors: scikkk
LastEditTime: 2022-11-14 13:34:18
Description: 主程序入口
'''

import argparse
import torch
import torch.nn as nn
from data import *
from model import *
from tqdm import tqdm
from utils import to_one_hot, plt_loss


class Loss(nn.Module):
    def __init__(self, decoder_type="AE") -> None:
        super().__init__()
        self.decoder_type = decoder_type

    def recon_loss(self, x, x_):
        # 实现重构损失函数 (5/100)
        # 使用MSE
        mse = torch.nn.functional.mse_loss(x, x_, reduction='sum')
        return mse
        

    def kl_div(self, mu, sigma):
        # 实现kl散度的损失 (5/100)
        # 计算与标准正态分布的KL散度
        var = torch.pow(sigma, 2)
        kld = 0.5*torch.sum(-1 + var + torch.pow(mu, 2) - torch.log(var))
        return kld

    def forward(self, x, x_, mu=None, sigma=None):
        # 实现loss的计算 (10/100)
        if self.decoder_type == "AE":
            return self.recon_loss(x, x_)
        elif self.decoder_type in ["VAE", "CVAE"]:
            mse = self.recon_loss(x, x_)
            kld = self.kl_div(mu, sigma)
            return mse + kld
        


def train(model, loader, loss, optimizer, epoch_num, ve_type):
    loss_record, iter_num = [], []
    all_pbar = tqdm(range(epoch_num), position=1)
    for epoch in all_pbar:
        pbar = tqdm(loader, position=0, leave=False, desc=f"Epoch {epoch}")
        for (x, y) in pbar:
            # 训练过程的补全 (20/100)
            x = x.view(x.shape[0], 1 * 28 * 28)
            optimizer.zero_grad()
            if ve_type == "AE":
                z, reco_x = model(x)
                L = loss.forward(x, reco_x)
            elif ve_type == "VAE":
                reco_x, z, mu, sigma = model(x)
                L = loss.forward(x, reco_x, mu, sigma)
            elif ve_type == "CVAE":
                y = to_one_hot(y.reshape(-1, 1), num_class=10)
                reco_x, z, mu, sigma = model(x, y)
                L = loss.forward(x, reco_x, mu, sigma)
            # BackPropagation
            L.backward()
            optimizer.step()
        
        with torch.no_grad():
            # 记录每一步的loss，用于后续分析
            loss_record.append(float(L))
            iter_num.append(epoch*len(loader))
            all_pbar.set_description(f"loss={float(L)}")
            # 保存一些重构出来的图像用于(写报告)进一步研究 (5/100)
            if(epoch == 0):
                # 保存数据集中的原始图像
                save_image(x.view(x.shape[0], 1, 28, 28), f"./results/{ve_type}/epoch_raw.png", nrow=10)
            # 保存每一步训练重构的图像 
            save_image(reco_x.view(x.shape[0], 1, 28, 28), f"./results/{ve_type}/epoch_{epoch}.png", nrow=10)
            if ve_type in ["VAE", "CVAE"]:
                # 保存自编码器生成的图像
                reco_x = model.generate()
                save_image(reco_x.view(x.shape[0], 1, 28, 28), f"./results/{ve_type}/gen_epoch_{epoch}.png", nrow=10)
    # 绘制loss折线图
    plt_loss(ve_type, loss_record, iter_num)
    torch.save(model.state_dict(), f"./models/{ve_type}.pth")



def main(args):
    encoder_args = {
        "x_dim": args.x_dim,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "decoder_type": args.ve_type,
    }
    decoder_args = {
        "x_dim": args.x_dim,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "decoder_type": args.ve_type,
    }
    encoder = Encoder(**encoder_args)
    decoder = Decoder(**decoder_args)
    ae = {"AE": AutoEncoder, "VAE": VariationalAutoEncoder, "CVAE": ConditionalVariationalAutoEncoder}
    auto_encoder = ae[args.ve_type](encoder, decoder)
    # 挑选你喜欢的优化器 :)
    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=args.lr)
    train_loader = get_data(train=True, batch_size=args.batch_size)
    loss = Loss(args.ve_type)
    train(model=auto_encoder, loss=loss, loader=train_loader, optimizer=optimizer, epoch_num=args.epoch_num, ve_type=args.ve_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ve_type", default="CVAE", choices=["AE", "VAE", "CVAE"])
    parser.add_argument("--x_dim", default=784, type=int)
    parser.add_argument("--latent_size", default=10, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--epoch_num", default=5, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    main(parser.parse_args())
