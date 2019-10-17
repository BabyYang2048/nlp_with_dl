import torch
import pickle
from torch.utils.data import Dataset, DataLoader

#dataSet需要自己实现，重写两个方法   __getitem__(self, item) &&  __len__(self):

vocab_data_path = "../data/vocab.pkl"
train_data_path = "../data/train_data.pkl"
valid_data_path = "../data/valid_data.pkl"


class DiseaseQuestion(Dataset):

    #构造方法 this，data ，字典 ，最大长度（一次数据几句话）  batch_size
    def __init__(self, data, vocab, max_len=12):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def pad_and_clip(self, sen):  # 句子预处理，填充和截断
        pad_len = self.max_len - len(sen)
        if pad_len > 0:
            # print("*****")
            # print("*" * 5)
            # [0] * 5 => [0,0,0,0,0]
            # sen = [1,2,3,4] +[0,0] = [1,2,3,4,0,0]
            # append => [1,2,3,4,[0,0]]
            return sen + [self.vocab.get("PAD")] * pad_len
        else:
            #[1:3],[1:-1],[1:]
            return sen[:self.max_len]

    def trans_sent2id(self, sen):
        # res = []
        # for c in sen:
        #     res.append(self.vocab.get(c))
        # return res
        return [self.vocab.get(c) for c in sen]

    # pytorch  所有的数据流通都需要转化为tensor才能传递，必须转换成long那一块
    def __getitem__(self, item):  #一次返回一条数据
        sen1, sen2, label, _ = self.data[item]  #下划线是我不需要的数据
        sen1 = torch.tensor(self.pad_and_clip(self.trans_sent2id(sen1))).long()
        sen2 = torch.tensor(self.pad_and_clip(self.trans_sent2id(sen2))).long()
        return sen1, sen2, torch.tensor(label)

    def __len__(self):    #返回dataset一条多长
        return len(self.data)


# 默认 64
def get_dataloader(batch_size=64):
    train_data = pickle.load(open(train_data_path, "rb"))
    valid_data = pickle.load(open(valid_data_path, "rb"))
    vocab_data = pickle.load(open(vocab_data_path, "rb"))
    ## todo
    train_loader = DataLoader(DiseaseQuestion(train_data, vocab_data),
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(DiseaseQuestion(valid_data, vocab_data),
                              batch_size=batch_size, shuffle=False)
    ## todo
    return train_loader, valid_loader
