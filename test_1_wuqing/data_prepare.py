import random
import math
import pickle

data = []
pos_data = []
neg_data = []
word2idx = {'PAD': 0, 'UNK': 1}

with open("../data/d_result.txt", encoding='utf-8') as f:
    for line in f:
        list = line.split("||")
        sen1 = list[0]
        sen2 = list[1]
        label = list[2]
        # print(sen1,sen2,label)
        for ch in sen1+sen2:
            if ch not in word2idx.keys():
                word2idx[ch] = len(word2idx)
        if int(label) == 0:
            neg_data.append((sen1, sen2, 0))
        else:
            pos_data.append((sen1, sen2, 1))
        data.append((sen1, sen2, int(label)))

split_size = 0.7

random.shuffle(neg_data)
random.shuffle(pos_data)

train_data = neg_data[:math.floor(split_size * len(neg_data))] + pos_data[:math.floor(split_size * len(neg_data))]
valid_data = neg_data[math.floor(split_size * len(neg_data)):] + pos_data[math.floor(split_size * len(neg_data)):]

pickle.dump(train_data,open("../data/train_data_BY.pkl","wb"))
pickle.dump(valid_data,open("../data/valid_data_BY.pkl","wb"))
pickle.dump(word2idx,open("../data/vocab_BY.pkl","wb"))