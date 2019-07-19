import numpy as np
import cv2

# 求欧式距离
def euclidean_distance(data1, data2):
    return np.sqrt(np.sum((data1 - data2) ** 2))


# gaussian penalty
def gaussian_penalty(x, sigma):
    return np.exp(-((x ** 2) / (2 * (sigma ** 2))))


# chi-square
def chi_square(his_1, his_2):
    return (1 / 2) * np.sum(((his_1 - his_2) ** 2)/(his_1 + his_2))

# 求直方图相关性
def hist_correlation(his_1, his_2):
    return cv2.compareHist(his_1, his_2, cv2.HISTCMP_CORREL)


def gaussian_weight(k, n_cicle):
    return np.exp(-((k ** 2) / ((2*n_cicle+1) ** 2)))

# 测试
if __name__ == "__main__":
    his1 = np.array([1, 2, 3, 4, 5])
    his2 = np.array([1, 1, 1, 1, 4])
    print(chi_square(his1, his2))