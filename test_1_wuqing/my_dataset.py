import torch
from torch.utils.data import Dataset, DataLoader
import pickle

vocab_data_path = "../data/vocab_BY.pkl"
train_data_path = "../data/train_data_BY.pkl"
valid_data_path = "../data/valid_data_BY.pkl"


class DiseaseQuestion(Dataset):
    def __init__(self, data, vocab, max_len=15):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def pad_and_clip(self, sen):
        pad_len = self.max_len - len(sen)
        if pad_len > 0:
            return sen + [self.vocab.get("PAD")] * pad_len
        else:
            return sen[:self.max_len]

    def trans_sent2id(self, sen):
        return [self.vocab.get(c) for c in sen]

    def __getitem__(self, item):
        sen1, sen2, label = self.data[item]
        sen1 = torch.tensor(self.pad_and_clip(self.trans_sent2id(sen1))).long()
        sen2 = torch.tensor(self.pad_and_clip(self.trans_sent2id(sen2))).long()
        return sen1, sen2, torch.tensor(label)

    def __len__(self):
        return len(self.data)


# 默认 64
def get_dataloader(batch_size=64):
    train_data = pickle.load(open(train_data_path, "rb"))
    valid_data = pickle.load(open(valid_data_path, "rb"))
    vocab_data = pickle.load(open(vocab_data_path, "rb"))
    train_loader = DataLoader(DiseaseQuestion(train_data, vocab_data),
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(DiseaseQuestion(valid_data, vocab_data),
                              batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader
