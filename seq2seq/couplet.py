from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import time
import math
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#########################################################################################################
# 数据处理
#########################################################################################################

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3
MAX_LENGTH = 15
first_txt_path = "../data/first.txt"
second_txt_path = "../data/second.txt"


class Vocab:
    """
        定义词典映射
    """

    def __init__(self, name):
        self.name = name        #上下文的字典拆开
        self.word2index = {}
        self.word2count = {}    #统计词频
        self.index2word = {0: "PAD", 1: "UNK", 2: "SOS", 3: "EOS"}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __len__(self):
        return len(self.index2word)


def read_data(first_txt, second_txt):
    """
        读取无情对数据并创建词典
    :param first_txt:       上联
    :param second_txt:      下联
    :return:    input_vocab, output_vocab, pairs = 上联词典，下联词典， 对联对
    """
    # 按照文件读取无情对数据
    first_lines = open(first_txt, encoding='utf-8').read().strip().split('\n')
    second_lines = open(second_txt, encoding='utf-8').read().strip().split("\n")

    # 合并数据!!!(cool)
    pairs = [[f, s] for f, s in zip(first_lines, second_lines)]

    # 创建对应词典
    input_vocab = Vocab("上联")
    output_vocab = Vocab("下联")

    return input_vocab, output_vocab, pairs


def filter_pair(p):
    """
        根据最大长度，过滤对联
    :param p: 一副对联
    :return:  True or False
    """
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    """
        过滤全部的对联
    :param pairs:   全部对联
    :return:    长度过滤后的对联
    """
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2):
    """
        处理数据，读取，过滤，创建词典等
    :param lang1:   上联文件路径
    :param lang2:   下联文件路径
    :return:    上联词典， 下联词典， 全部对联
    """
    input_lang, output_lang, pairs = read_data(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# 处理数据
ipt_vocab, opt_vocab, sen_pairs = prepare_data(first_txt_path, second_txt_path)
print(random.choice(sen_pairs))

#########################################################################################################
# 定义模型
#########################################################################################################


class EncoderRNN(nn.Module):
    """
        RNN-based Encoder
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size  # 隐层维度
        self.embedding = nn.Embedding(input_size, hidden_size)  # 上联嵌入层
        self.gru = nn.GRU(hidden_size, hidden_size)     # GRU

    def forward(self, input, hidden): #-1 是自己去算的
        embedded = self.embedding(input).view(1, 1, -1)     # 调整数据维度为 (1, 1, hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)     # GRU编码
        return output, hidden

    def init_hidden(self):  #初始化隐藏状态
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    """
        RNN-based Decoder with Attention
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size  # 隐层维度
        self.output_size = output_size  # 输出维度
        self.dropout_p = dropout_p  # dropout_rate
        self.max_length = max_length    # 最大文本长度

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)   # 下联嵌入层（词典大小，隐藏层维度）
        # Attention Module  现在的attention就是两个全连接
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)   # Dropout 防止过拟合
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)   # GRU
        self.out = nn.Linear(self.hidden_size, self.output_size)    # 输出层

    def forward(self, input, hidden, encoder_outputs):

        embedded = self.embedding(input).view(1, 1, -1)     # 调整数据维度为 (1, 1, hidden_size)
        embedded = self.dropout(embedded)   # Dropout

        # 计算注意力权重
        # atten_weights 是15维！！！
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 加权求和！！！
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # 拼接
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # 映射
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)   # GRU解码

        output = F.log_softmax(self.out(output[0]), dim=1)  # 预测结果
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


#########################################################################################################
# 数据转换 -> tensor
#########################################################################################################

def indexes_from_sentence(lang, sentence):
    """
            将句子中的字转换为索引
    :param lang:    词典
    :param sentence:    句子
    :return:    索引构成的句子
    """
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    """
        将句子转换为索引，增加终止符号后转换为tensor
    :param lang:    词典
    :param sentence:    句子
    :return:    tensor
    """
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair):
    """
        转换一副无情对为tensor
    :param pair:    一副无情对
    :return:    tensor
    """
    input_tensor = tensor_from_sentence(ipt_vocab, pair[0])
    target_tensor = tensor_from_sentence(opt_vocab, pair[1])
    return (input_tensor, target_tensor)


#########################################################################################################
# 时间格式化   为了打印输出 非重点
#########################################################################################################

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


#########################################################################################################
# 训练过程
#########################################################################################################


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    """
        单次训练流程
    :param input_tensor:    上联构成的tensor
    :param target_tensor:   下联构成的tensor
    :param encoder:     编码器
    :param decoder:     解码器
    :param encoder_optimizer:   编码器的优化器
    :param decoder_optimizer:   解码器的优化器
    :param criterion:   损失函数
    :param max_length:  最大文本长度
    :return:    损失值
    """
    encoder_hidden = encoder.init_hidden()      # 初始化编码器的初始状态

    encoder_optimizer.zero_grad()   # 编码器梯度归零
    decoder_optimizer.zero_grad()   # 解码器梯度归零

    input_length = input_tensor.size(0)     # 上联长度
    target_length = target_tensor.size(0)   # 下联长度

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)   # 存储输出， (max_length, hidden_size)

    loss = 0    # 记录损失

    for ei in range(input_length):      # 根据上联长度遍历数据
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)      # 编码
        encoder_outputs[ei] = encoder_output[0, 0]  # 存储输出至encoder_outputs

    decoder_input = torch.tensor([[SOS_token]], device=device)      # 设置Decoder输入，SOS表示开始解码

    decoder_hidden = encoder_hidden     # 将解码器初始的hidden设置为编码器的最终hidden

    teacher_forcing_ratio = 0.5  # 强制学习的概率
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False    # 判断是否需要强制学习

    if use_teacher_forcing:
        # 将目标字作为下次预测的输入
        for di in range(target_length):     # 循环解码
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])    # 计算损失
            decoder_input = target_tensor[di]  # 强制替换
    else:
        # 将预测结果作为下次的输入
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)     # 取top1的结果
            decoder_input = topi.squeeze().detach()  # 取出预测结果
            loss += criterion(decoder_output, target_tensor[di])    # 计算损失
            if decoder_input.item() == EOS_token:   # 如果读取到EOS则终止解码
                break

    loss.backward()     # 损失反向传播

    encoder_optimizer.step()    # 更新编码器参数
    decoder_optimizer.step()    # 更新解码器参数

    return loss.item() / target_length


def train_iters(encoder, decoder, n_iters, print_every=100, plot_every=100, learning_rate=0.001):
    """
        迭代训练
    :param encoder:     编码器
    :param decoder:     解码器
    :param n_iters:     迭代轮数
    :param print_every:     日志频率
    :param plot_every:      损失绘制频率
    :param learning_rate:   学习率
    :return:    None
    """
    start = time.time()     # 记录时间
    plot_losses = []        # 记录损失
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)      # 定义编码器优化器
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)      # 定义解码器优化器
    training_pairs = [tensors_from_pair(random.choice(sen_pairs)) for i in range(n_iters)]  # 生成n_iters条数据
    criterion = nn.NLLLoss()    # 定义损失函数

    for iter in range(1, n_iters + 1):      # 迭代训练
        training_pair = training_pairs[iter - 1]    # 取数据
        input_tensor = training_pair[0]     # 上联
        target_tensor = training_pair[1]    # 下联

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)  # 训练，得到损失
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:     # 迭代至打印次数，打印日志
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:      # 迭代至绘图次数，记录损失
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)      # 绘制损失图像


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


#########################################################################################################
# 评估模型
#########################################################################################################


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    """
        评估结果
    :param encoder:     编码器
    :param decoder:     解码器
    :param sentence:    预测句子
    :param max_length:  最大文本长度
    :return:    结果
    """
    with torch.no_grad():   # 不需要计算梯度
        input_tensor = tensor_from_sentence(ipt_vocab, sentence)    # 将上联转换为tensor
        input_length = input_tensor.size()[0]   # 得到文本长度
        encoder_hidden = encoder.init_hidden()  # 初始化编码器隐层

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)   # 记录编码器输出

        for ei in range(input_length):      # 同上
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        # 解码阶段不再需要判断是否需要强制学习
        # 没有target标签，无法强制学习
        # 直接选择预测结果作为下个输入
        for di in range(max_length):    # 同上
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(opt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(encoder, decoder, n=10):
    """
        随机评估一些数据
    :param encoder:     编码器
    :param decoder:     解码器
    :param n:   数据大小
    :return:    None
    """
    for i in range(n):
        pair = random.choice(sen_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    hidden_size = 64    # 隐层维度
    encoder1 = EncoderRNN(ipt_vocab.n_words, hidden_size).to(device)    # 编码器
    attn_decoder1 = AttnDecoderRNN(hidden_size, opt_vocab.n_words, dropout_p=0.1).to(device)    # 解码器
    train_iters(encoder1, attn_decoder1, 50000, print_every=1000)     # 开始训练
    evaluate_randomly(encoder1, attn_decoder1)  # 评估
