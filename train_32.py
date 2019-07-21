import numpy as np
from network.outliers_removal_network import ORNet
# 数据
load_labeled = np.load("./data/train_data/32bit/train_data.npz")
train_descriptor = load_labeled["train_descriptor"]
train_label = load_labeled["train_label"]
# 声明网络
ornet = ORNet(None)
for i in range(50000):
    index = np.random.randint(0, 3300, 512)
    # 训练
    loss = ornet.train(train_descriptor[index], train_label[index])
    print(loss)
ornet.save()

