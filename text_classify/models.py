import torch
import torch.nn as nn
import torch.nn.functional as F


class Cnn_model(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # self.cnn_5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=1)
        self.cnn_3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3)
        self.cnn_1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # (batch,hidden,len)->(batch,len,hidden)
        x = x.transpose(1, 2) #维度转换
        # cnn_5 = self.drop(F.relu(self.cnn_5(x))).mean(dim=-1).squeeze(-1)
        cnn_3 = self.drop(F.relu(self.cnn_3(x))).mean(dim=-1).squeeze(-1)
        cnn_1 = self.drop(F.relu(self.cnn_1(x))).mean(dim=-1).squeeze(-1)
        # cnn_1 = (b,hidden) cnn_2 = (b,hidden)
        cnn_enc = torch.cat([cnn_3, cnn_1], dim=-1)
        # cnn_enc = (b,hidden*2)
        return cnn_enc


class BaseClassification(nn.Module):

    def __init__(self, vocab_size, hidden_dim, mode=None):
        super().__init__()
        #随机 初始化 词向量
        #embedding = ( v * h )
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        #防止过拟合，（随机把一些东西设置为0，只在训练时使用）
        self.drop = nn.Dropout(0.3)
        self.mode = mode.lower()
        if mode.lower() == 'lstm':
            self.encode_layer = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                                        batch_first=True, bidirectional=True)
        elif mode.lower() == "gru":
            self.encode_layer = nn.GRU(hidden_dim, hidden_dim, num_layers=1,
                                       batch_first=True, bidirectional=True)
        elif mode.lower() == "cnn":
            self.encode_layer = Cnn_model(hidden_dim)
        else:
            self.encode_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), #全连接
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim * 2) #全连接（维度变化）
            )
        # 这里是输入维度  如果多加了层数需要在这里改
        self.predict_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim *2),  #二分类
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, 2)
        )

    '''
        关于维度：b-> batch_size
            x1 = (b*max_len)                    x2 = (b*max_len)
            
            x1 = ( b*max_len*hidden_dim)        x2 = ( b*max_len*hidden_dim)
            
            x1 = ( b*max_len*(hidden_dim*2))    x2 = ( b*max_len*(hidden_dim*2))
    '''

    def forward(self, x1, x2): #（x1第一句，x2第二句）
        x1, x2 = self.drop(self.embedding(x1)), self.drop(self.embedding(x2))
        #把词向量送进全连接 编码
        x1, x2 = self.encode_layer(x1), self.encode_layer(x2)
        # output , (hidden , state) = LSTM(x1)
        # output

        if self.mode in ["lstm", 'gru']:
            x1, x2 = x1[0], x2[0]
        if self.mode != 'cnn':
            x1, x2 = x1.mean(dim=1).squeeze(), x2.mean(dim=1).squeeze()
            # x1= (b,1,hidden_dim*2) -> x1 = (b,hidden_dim*2)
        final_enc = torch.cat([x1, x2], dim=-1)
        # 拼起来 final_enc = (b,hidden*4)  dim=-1 说明在最后一个维度上拼起来
        return self.predict_layer(final_enc)
        #[b,[0.3,0.7]]

    # X = x1,x2  Y' = [b,2]
