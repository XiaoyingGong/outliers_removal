import numpy as np


# 求欧式距离
def euclidean_distance(data1, data2):
    return np.sqrt(np.sum((data1 - data2) ** 2))


# gaussian penalty
def gaussian_penalty(x, sigma):
    return np.exp(-((x ** 2) / (2 * (sigma ** 2))))

# 测试
if __name__ == "__main__":
    a = 1
