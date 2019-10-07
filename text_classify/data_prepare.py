import pickle
import csv
import random
import math

data = []
neg_data = []
pos_data = []
c_dict = {'aids': 0, 'diabetes': 1, 'hypertension': 2, 'hepatitis': 3, 'breast_cancer': 4}
word2idx = {'PAD': 0, "UNK": 1}  # 我们要的字典，PAD是填充值，默认0  ，UNK是字典里没有的字，我们也需要映射，映射到unk上

with open("../data/train.csv", encoding='utf-8') as f:  #开始读数据
    csv_data = list(csv.reader(f))  #语法：  读csv数据转换为list
    for ii, d in enumerate(csv_data):  # #语法  for循环，enumerate是产生索引的方法，这里变量为ii
        if ii == 0:  #剔除掉第一行
            continue
        else:
            sen1, sen2, label, c = d
            for ch in sen1+sen2:
                if ch not in word2idx.keys():
                    word2idx[ch] = len(word2idx)
            if c not in c_dict:
                c_dict[c] = len(c_dict)
            if int(label) == 0:         #负例  (sen1,sen2,0,c)
                neg_data.append((sen1.replace("\"", ""), sen2.replace("\"", ""), 0, c_dict.get(c)))
            else:                       #正例  (sen1,sen2,1,c)
                pos_data.append((sen1.replace("\"", ""), sen2.replace("\"", ""), 1, c_dict.get(c)))

            data.append((sen1.replace("\"", ""), sen2.replace("\"", ""), int(label)))
#训练集和验证集（7:3）分
split_size = 0.7
#shuffle是打乱顺序
random.shuffle(neg_data)
random.shuffle(pos_data)
#正例和负例占比相同
train_data = neg_data[:math.floor(split_size * len(neg_data))] + pos_data[:math.floor(split_size * len(neg_data))]
valid_data = neg_data[math.floor(split_size * len(neg_data)):] + pos_data[math.floor(split_size * len(neg_data)):]
#这里应该再shuffle一次

print(c_dict)
#pickle是做 序列化的，用法pickle.dump  对象，文件（以二进制形式存）
pickle.dump(train_data, open("../data/train_data.pkl", "wb"))
pickle.dump(valid_data, open("../data/valid_data.pkl", "wb"))
pickle.dump(word2idx, open("../data/vocab.pkl", 'wb'))


###如果想让验证更好的话，使用交叉验证
###把数据集分为5份，1,4划分验证集和训练集然后依次跑5个模型，最后模型融合