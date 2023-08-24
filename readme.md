# 实验报告



## 1 实验内容

实现三种自编码器 包括 `AutoEncoder`, `Variational AutoEncoder`, `Conditional Variational AutoEncoder` .

报告中重建的数字都以下图作为输入：

<img src=".\results\CVAE\epoch_raw.png" alt="epoch_raw" style="zoom:80%;" />



## 2 实验原理

### 2.1 AE

`AutoEncoder` 提出来主要是为了解决 **数据压缩与还原**的问题, 我们输入一个 `x`, 通过 `encoder` 得到隐变量 `z`, 再将 ` z`  经过 `decoder` 的处理还原出 `x'`. 于是很自然的, 优化目标就应该是希望 $\textbf{dist}(x,x')$ 尽量小.

### 2.2 VAE

`Variational AutoEncoder` 提出来是为了做 **生成式的任务**, 希望模型可以通过学习, 获得如何生成和输入的 `x` 很像但是不完全一样的 `x'`, 比如我们希望生成动漫头像, 我们当然是希望生成的头像与训练的数据是风格相似但是不完全一样的. 为了做到这一点, 我们采取变分推断的方式, 不再是通过 `encoder` 获得一个隐变量, 而是生成这个隐变量服从的分布的均值与方差(实际上这里我们假设隐变量服从高斯分布, 那么均值和方差实际上就唯一刻画了这个分布的所有信息). 再经过 `decoder` 的处理得到 `x'`, 此时的优化目标除了让 $\textbf{dist}(x,x')$ 尽可能小以外, 还应该满足对隐变量获得的分布的约束 $\textbf{KL}(P_z, prior_z)$ , 也就是我们希望我们的 `z` 服从的分布与我们给定的先验分布(这里是高斯分布)也尽量接近.

### 2.3 CVAE

`Conditional Variational AutoEncoder` 则是在 `VAE` 的基础上, 加入了对数据类别信息的指定, 从而我们可以做到指定模型生成具体某个类别数据。

具体而言 `VAE` 的输入只是 `x`，而 `CAVE` 的输入是 `(x,y)` .



## 3  实验细节

### 3.1 损失函数选取

```python
mse = torch.nn.functional.mse_loss(x, x_, reduction='sum')
```

为了衡量重建前后数字的差异，选取 `mse_loss`，但是发现对于 `VAE` 效果较差，修改 `ruduction` 方式为 `sum`，即不取平均，发现有明显改善。二者对比图如下，均使用默认参数，例图为训练5个 `epoch` 后的结果：

<img src=".\image\loss.png" alt="loss" style="zoom:80%;" />

### 3.2 CVAE网络结构调整

```markdown
# Encoder 
    - self.mu
        nn.Linear(x_dim+label_size, hidden_size)
        nn.ReLU()
        nn.Linear(hidden_size, latent_size)
        nn.ReLU()
    - self.sigma
        nn.Linear(x_dim+label_size, hidden_size)
        nn.ReLU()
        nn.Linear(hidden_size, latent_size)
# Decoder
    nn.Linear(latent_size+label_size, hidden_size)
    nn.ReLU()
    nn.Linear(hidden_size, x_dim)
```

使用默认的网络结构训练，但是发现对于 `CVAE` 效果较差，对编码器中的 `self.mu` 网络增加激活函数 `ReLU`，，发现效果有明显改善。二者对比图如下，均使用默认参数，例图为训练5个 `epoch` 后的重建结果与指定标签的结果：

<img src=".\image\CVAE_ReLU.png" alt="CVAE_ReLU" style="zoom:80%;" />

## 4 超参数设置

调节参数时为节约时间，且考虑到本任务比较简单，只训练5个 `epoch` ，然后对比所得结果。

### 4.1 AE

先调节 `hidden_size`，结果对比图如下：

<img src=".\image\AE_hiddensize.png" alt="AE_hiddensize" style="zoom:80%;" />

发现 `hidden_size=256` 较好，再调节训练率`lr`，发现选择 `lr=0.001` 较好，对比图如下：

<img src=".\image\AE_lr.png" alt="AE_lr" style="zoom:80%;" />

### 4.2 VAE

先调节粗调 `hidden_size`，对比图如下：

<img src=".\image\VAE_hiddensize_1.png" alt="VAE_hiddensize_1" style="zoom:80%;" />

发现取 `100~200` 较好，再细调 `hidden_size`：

<img src=".\image\VAE_hiddensize_2.png" alt="VAE_hiddensize_2" style="zoom:80%;" />

发现 `hidden_size=120` 较好，再调节训练率`lr`，发现选择 `lr=0.01` 较好，对比图如下：

<img src=".\image\VAE_lr.png" alt="VAE_lr" style="zoom:80%;" />

### 4.3 CVAE

先调节`hidden_size`，对比图如下：

<img src=".\image\CVAE_hiddensize.png" alt="CVAE_hiddensize" style="zoom:80%;" />

发现 `hidden_size=200` 较好，再调节训练率`lr`，发现选择 `lr=0.01` 较好，对比图如下：

<img src=".\image\CVAE_lr.png" alt="CVAE_lr" style="zoom:80%;" />



## 5 实验结果

### 5.1 AE

优化器选择`Adam`，其它参数如下：

```shell
--latent_size=10, --hidden_size=256, --batch_size=100, --lr=0.001
```

#### 5.1.1 Train Loss

<img src=".\results\AE\AE_256_001.png" alt="AE_256_001" style="zoom:80%;" />

#### 5.1.2 图像重构对比

左图为输入，右图为输出

<img src=".\results\AE\epoch_raw.png" alt="epoch_raw" style="zoom:80%;" />	<img src=".\results\AE\epoch_31.png" alt="epoch_reconstruction" style="zoom:80%;" />

### 5.2 VAE

优化器选择`Adam`，其它参数如下：

```shell
--latent_size=10, --hidden_size=120, --batch_size=100, --lr=0.01
```

#### 5.2.1 Train Loss

<img src=".\results\VAE\VAE_120_01.png" alt="VAE_120_01" style="zoom:80%;" />

#### 5.2.2 图像重构对比

左图为输入，右图为输出：

<img src=".\results\VAE\epoch_raw.png" alt="epoch_raw" style="zoom:80%;" />	<img src=".\results\VAE\epoch_31.png" alt="epoch_31" style="zoom:80%;" />

#### 5.2.3 随机采样

从$\mathcal{N}(0,1)$ 中采样一些点作为隐变量, 生成一些图片查看结果，下图是两次运行的结果：

<img src=".\results\VAE\gen_epoch_12.png" alt="gen_epoch_12" style="zoom:80%;" />	<img src=".\results\VAE\gen_epoch_31.png" alt="gen_epoch_31" style="zoom:80%;" />

####  5.2.4 均匀采样

将 `latent_size` 改为2，从以原点为中心，15为边长的正方形内均匀采样，作为Decoder的输入。

```shell
--latent_size=2, --hidden_size=120, --batch_size=100, --lr=0.01
```
先看缩略图，发现不同数字大概呈放射状分布：

<img src=".\results\VAE_grid\grid_0.png" alt="grid_0" style="zoom: 10%;" />		<img src=".\results\VAE_grid\grid_1.png" alt="grid_1" style="zoom:10%;" />		<img src=".\results\VAE_grid\grid_2.png" alt="grid_2" style="zoom:10%;" /> 

放大图片观察细节，发现每个扇形内部的数字比较清晰，但是若采样采到扇形的边界交汇处，会导致生成不规范的数字：

<img src=".\results\VAE_grid\grid_3.png" alt="grid_3" style="zoom:60%;" />

### 5.3 CVAE

优化器选择`Adam`，其它参数如下：

```shell
--latent_size=10, --hidden_size=200, --batch_size=100, --lr=0.01
```

#### 5.3.1 Train Loss

<img src=".\results\CVAE\CVAE_200_01.png" alt="CVAE_200_01" style="zoom:80%;" />

#### 5.3.2 图像重构对比

左图为输入，右图为输出：

<img src=".\results\CVAE\epoch_raw.png" alt="epoch_raw" style="zoom:80%;" />	<img src=".\results\CVAE\epoch_31.png" alt="epoch_31" style="zoom:80%;" />

#### 5.3.2 指派标签效果

指派具体的标签, 从 $\mathcal{N}(0,1)$ 中采样, 生成一些图片查看结果，下图是两次运行的结果：

<img src=".\results\CVAE\gen_epoch_0.png" alt="gen_epoch_0" style="zoom:80%;" />	<img src=".\results\CVAE\gen_epoch_28.png" alt="gen_epoch_28" style="zoom:80%;" />



## 6 总结与思考

1. 观察 `AE`, `VAE`, `CVAE` 的结果，发现 `4` 和 `5` 的学习效果最差，其中习得的 `4` 右边几乎不会出头，而习得的 `5` 和 `S` 较为相似，但是观察数据集，发现数据集中确实有这种特征。要想取得更好的训练效果，只调整参数可能较为困难，需要从数据集的采样方式入手。

   <img src=".\image\4.png" alt="4" style="zoom: 30%;" /> <img src=".\image\5.png" alt="5" style="zoom: 30%;" />

2. `VAE` 采样生成的数字（下图左）会含有介于两种数字直接的情况，因为随机采样难免会采样到两种数字的过渡区域；而 `CVAE` 采样生成的数字（下图右）不会出现这种情况。应该是增加的标记维度使不同的数字被分隔开，从而容易取得较好的效果：

   <img src=".\results\VAE\gen_epoch_31.png" alt="gen_epoch_31" style="zoom:80%;" /><img src=".\results\CVAE\gen_epoch_28.png" alt="gen_epoch_28" style="zoom:80%;" />
