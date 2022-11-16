'''
Author: scikkk 203536673@qq.com
Date: 2022-11-12 22:41:32
LastEditors: scikkk
LastEditTime: 2022-11-14 13:01:50
Description: some tools
'''

import matplotlib.pyplot as plt
import torch
from math import log2

def plt_loss(ve_type:str, loss:list, iter_n:list) -> None:
    """
    用于绘制损失函数下降折线图
    :param ve_type: 用于分辨自编码器类型
    :param loss: 每次训练后的loss
    :param itern: loss对应的迭代次数 
    """
    #准备绘制数据
    # print(loss)
    # print(iter_n)
    y = [log2(i) for i in loss]
    # y = loss
    x = list(range(len(y)))
    for i in x[1:]:
        plt.plot(iter_n[i-1:i+1], y[i-1:i+1], "deepskyblue" if (y[i] <= y[i-1]) else "red")
    #绘制坐标轴标签
    plt.xlabel("Iter num")
    plt.ylabel("Loss")
    plt.title(ve_type)
    #保存图片
    plt.savefig(f"./results/{ve_type}/{ve_type}.png")
    plt.show()

def to_one_hot(labels: torch.Tensor, num_class=10) -> torch.Tensor:
    """
    用于将图片标记转化为独热编码
    :param labels: 标记
    :param num_class: 类别总数，也即独热码长度
    """
    y = torch.zeros(labels.shape[0], num_class)
    for i, label in enumerate(labels):
        y[i, label] = 1
    return y