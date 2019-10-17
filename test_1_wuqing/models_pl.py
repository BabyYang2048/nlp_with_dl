import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from test_1_wuqing.my_dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_to_device(data):
    for d in data:
        yield d.to(device)


class BaseClassificationModel(pl.LightningModule):

    def __init__(self, mode):
        super(BaseClassificationModel, self).__init__()
        vocab_size = len(self.train_dataloader().dataset.vocab)
        hidden_dim = 128
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.drop = torch.nn.Dropout(0.3)
        self.mode = mode.lower()
        if mode.lower() == "lstm":
            self.encode_layer = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        elif mode.lower() == "gru":
            self.encode_layer = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.encode_layer = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),  # 全连接
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim * 2)  # 全连接（维度变化）
            )
        self.predict_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim * 2, 2),
        )

    def forward(self, x1, x2):
        x1, x2 = self.drop(self.embedding(x1)), self.drop(self.embedding(x2))
        x1, x2 = self.encode_layer(x1), self.encode_layer(x2)
        if self.mode in ["lstm", 'gru']:
            x1, x2 = x1[0], x2[0]
        if self.mode != 'cnn':
            x1, x2 = x1.mean(dim=1).squeeze(), x2.mean(dim=1).squeeze()
        final_enc = torch.cat([x1, x2], dim=-1)
        return self.predict_layer(final_enc)

    def training_step(self, batch, batch_nb):
        """
        返回损失，附带tqdm指标的dict
        :param batch:
        :param batch_nb:
        :return:
        """
        # REQUIRED

        # x, y = batch
        # y_hat = self.forward(x)  #y_hat对应为线性回归模型的预测值
        # return {'loss': F.cross_entropy(y_hat, y)}

        x1, x2, y = data_to_device(batch)
        y_hat = self.forward(x1, x2)
        return {'loss': F.cross_entropy(y_hat,y)}

    def validation_step(self, batch, batch_nb):
        """
        返回需要在validation_end中聚合的所有输出
        :param batch:
        :param batch_nb:
        :return:
        """
        # OPTIONAL
        # x, y = batch
        # y_hat = self.forward(x)
        # return {'val_loss': F.cross_entropy(y_hat, y)}
        acc_num = 0
        x1, x2, y = data_to_device(batch)
        y_hat = self.forward(x1,x2)
        # loss = torch.nn.CrossEntropyLoss(y_hat,y)
        acc_num += (torch.argmax(y_hat, dim=-1).long() == y.long()).sum().cpu().item()
        #这个是计算出的正确的个数吗？
        #然后我在end里在除法？
        return {'acc_num':acc_num ,'val_loss':F.cross_entropy(y_hat,y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # print(self.val_dataloader())
        acc = sum([x['acc_num'] for x in outputs])/len(self.val_dataloader()[0].dataset)
        print()
        print('avg_val_loss=',avg_loss.cpu().item(),'acc=',acc)

        return {'avg_val_loss': avg_loss}

    # def test_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}
    #
    # def test_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)

    @pl.data_loader
    def train_dataloader(self):
        # print("get_dataloader=",get_dataloader())
        train_loader,valid_loader = get_dataloader()
        # print("train_loader=", train_loader)
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        train_loader,valid_loader = get_dataloader()
        return valid_loader

    # @pl.data_loader
    # def test_dataloader(self):
    #     # OPTIONAL
    #     # can also return a list of test dataloaders
    #     return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=64)


model = BaseClassificationModel(mode="gru")
# most basic trainer, uses good defaults
trainer = Trainer(gpus='0', max_nb_epochs=15)
trainer.fit(model)

