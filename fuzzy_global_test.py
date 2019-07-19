import numpy as np

a = np.load("./data/exp/train_data.npz")
b = a["train_descriptor"][:, 16:]
c = a["train_label"]
k = 1
for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    if np.argmax(c[i]) == 0:
        print(k, ":匹配")
    else:
        print(k, ":不匹配")
    k += 1
    print(c[i])
    print(b[i])