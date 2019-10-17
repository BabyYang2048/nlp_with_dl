import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from pytorch_lightning import Trainer

import pytorch_lightning as pl

'''
以经典的 MNIST 图像识别为例，
如下展示了 LightningModel 的示例。
我们可以照常导入 PyTorch 模块，
但这次不是继承 nn.Module，
而是继承 LightningModel。
然后我们只需要照常写 PyTorch 就行了，
该调用函数还是继续调用。
这里看上去似乎没什么不同，
但注意方法名都是确定的，
这样才能利用 Lightning 的后续过程。
'''


class CoolModel(pl.LightningModule):

    def __init__(self):
        super(CoolModel, self).__init__()  #这是调用的pl的model
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10) #这个l1是什么？？

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))
        # x.view(x.size(0),-1) 这句话是说将x的输出拉伸为一行  -1是自适应的意思，x.size(0)是batch size

    def training_step(self, batch, batch_nb):
        """
        返回损失，附带tqdm指标的dict
        :param batch:
        :param batch_nb:
        :return:
        """
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)  #y_hat对应为线性回归模型的预测值
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        """
        返回需要在validation_end中聚合的所有输出
        :param batch:
        :param batch_nb:
        :return:
        """
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=0.001)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=64)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=64)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        # can also return a list of test dataloaders
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=64)


model = CoolModel()

# most basic trainer, uses good defaults
trainer = Trainer(gpus='1', max_nb_epochs=3)
trainer.fit(model)
