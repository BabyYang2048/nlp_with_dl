import torch
import time
import torch.nn as nn
from test_1_wuqing.models import BaseClassification
from test_1_wuqing.my_dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def data_to_device(data):
    for d in data:
        yield d.to(device)


def train(model, train_loader, valid_loader, loss_func, optimizer, epochs=5):
    best_score = 0.5
    for ep in range(epochs):
        print("EPOCH", ep)
        acc_num = 0
        losses = []
        start_time = time.time()
        for ii, d in enumerate(train_loader):
            x1, x2, label = data_to_device(d) #把数据取出来
            optimizer.zero_grad()  #整个网络梯度归0
            output = model(x1, x2) #得到y'
            loss = loss_func(output, label) #计算损失值
            loss.backward()  #反向传播
            optimizer.step() #更新权重
            losses.append(loss.cpu().item())
            acc_num += (torch.argmax(output, dim=-1).long() == label.long()).sum().cpu().item()
        print("EPOCH %d, TRAIN MEAN LOSS = %f, TRAIN ACCURACY = %f, SPEND TIME: %d" %
              (ep, sum(losses) / len(losses), acc_num / len(train_loader.dataset), time.time() - start_time))
        score = evaluate(model, valid_loader, ep)
        if score > best_score:
            best_score = score
            # torch.save(model.state_dict(), './model')
        print("BEST SCORE:", best_score)


def evaluate(model, valid_loader, ep):
    model.eval()
    with torch.no_grad():
        acc_num = 0
        losses = []
        for ii, d in enumerate(valid_loader):
            x1, x2, label = data_to_device(d)
            output = model(x1, x2)
            loss = loss_func(output, label)
            losses.append(loss.cpu().item())
            acc_num += (torch.argmax(output, dim=-1).long() == label.long()).sum().cpu().item()
        print("EPOCH %d, VALID MEAN LOSS = %f, VALID ACCURACY = %f" %
              (ep, sum(losses) / len(losses), acc_num / len(valid_loader.dataset)))
    model.train()
    return acc_num / len(valid_loader.dataset)


if __name__ == '__main__':
    # 加载数据
    print("加载数据")
    train_loader, valid_loader = get_dataloader()
    # 创建模型
    print("创建模型")
    model = BaseClassification(len(train_loader.dataset.vocab), 128, mode="gru")
    model.to(device)
    # 定义损失函数
    print("定义损失")
    loss_func = nn.CrossEntropyLoss()
    # 定义优化器
    print("定义优化器")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # 训练模型
    print("训练模型")
    train(model, train_loader, valid_loader, loss_func, optimizer, epochs=15)


