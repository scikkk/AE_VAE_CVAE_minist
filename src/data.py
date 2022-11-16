'''
Author: scikkk 203536673@qq.com
Date: 2022-11-09 15:49:16
LastEditors: scikkk
LastEditTime: 2022-11-14 00:11:16
Description: data loader
'''
import torch.utils.data as td
import torchvision.datasets.mnist as mnist
import torchvision.transforms as transforms
from torchvision.utils import save_image

def get_data(train=True, batch_size=128):
    dataset = mnist.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.1307,], [0.3081,],),
            ]
        ),
    )
    return td.DataLoader(dataset, batch_size=batch_size)

if __name__ == "__main__":
    loader = get_data()
    # print(len(loader))
    for i, (X, y) in enumerate(loader):
        if i > 0:
            break
        print(X)
        # print(y)
        print(X.shape)  # (batch_size, 1, 28, 28)
        print(y.shape) # (batch_size)
        save_image(X, './raw/{}.png'.format(i))
