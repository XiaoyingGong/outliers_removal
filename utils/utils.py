import numpy as np


# 求欧式距离
def euclidean_distance(data1, data2):
    return np.sqrt(np.sum((data1 - data2) ** 2))


# gaussian penalty
def gaussian_penalty():
    return np.exp()

# 测试
if __name__ == "__main__":
    a = np.array([3])
    b = np.array([4])
    print(euclidean_distance(a, b))
