import numpy as np
import cv2

# 求欧式距离
def euclidean_distance(data1, data2):
    return np.sqrt(np.sum((data1 - data2) ** 2))

# 将向量标准化后的欧氏距离
def euclidean_distance_standard_vector(data1, data2):
    data1 = data1 / np.sqrt(np.sum(data1**2) + 0.0000000000000001)
    data2 = data2 / np.sqrt(np.sum(data2**2) + 0.0000000000000001)
    return euclidean_distance(data1, data2)

# 将结果归一化后的欧式距离
def euclidean_distance_normalized(data_1, data_2):
    return 1 / (euclidean_distance(data_1, data_2) + 1)

# gaussian penalty
def gaussian_penalty(x, sigma):
    return np.exp(-((x ** 2) / (2 * (sigma ** 2))))


# chi-square
def chi_square(his_1, his_2):
    return (1 / 2) * np.sum(((his_1 - his_2) ** 2)/(his_1 + his_2))

# 求直方图相关性
def hist_correlation(his_1, his_2):
    return cv2.compareHist(his_1, his_2, cv2.HISTCMP_CORREL)

# 在模糊计数中，求高斯的数目
def gaussian_weight(current_circle_index, count_circle_index, circles_num):
    return np.exp(-(current_circle_index - count_circle_index)**2 / (circles_num**2))



def normalized_euclidean(a, b):
    sumnum = 0
    for i in range(len(a)):
        avg = (a[i] - b[i]) / 2
        si = ((a[i] - avg) ** 2 + (b[i] - avg) ** 2) ** 0.5
        sumnum += ((a[i] - b[i]) / si) ** 2
    return sumnum ** 0.5

# 测试
if __name__ == "__main__":
    data_1 = np.array([99, 99])
    data_2 = np.array([500, 88])
    print(euclidean_distance(data_1, data_2))
    print(euclidean_distance_normalized(data_1, data_2))


