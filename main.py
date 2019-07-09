import matplotlib.pyplot as plt
import cv2
import numpy as np
from feature_matching import sift_matching
from knn.knn import K_NearestNeighbors
from match_descriptor.angle_sift import AngleSift
from  match_descriptor.fuzzy_global_circle import FuzzyGlobalCircle
# 主类，汇总各个类的功能

# 图像路径
img1_path = "./img/1_r.png"
img2_path = "./img/1_s.png"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
h_img = np.hstack((img1, img2))

sift_threshold = 0.9

# 通过sift进行预匹配
pre_matches1, pre_matches2, des1, des2, match_index = sift_matching.get_matches(img1, img2, sift_threshold)
# 因为匹配里面也有可能存在一对多的情况所以，这里进行一次将一对多的情况剔除
pre_matches1, index1 = np.unique(pre_matches1, return_index=True, axis=0)
pre_matches2 = pre_matches2[index1]
pre_matches2, index2 = np.unique(pre_matches2, return_index=True, axis=0)
pre_matches1 = pre_matches1[index2]

len1 = len(pre_matches1)
len2 = len(pre_matches2)

for i in [69]:
    pointIndex = i# 69
    # 将prematch转置，便于matplotlib绘制
    pre_matches1_t = np.transpose(pre_matches1)
    pre_matches2_t = np.transpose(pre_matches2)

    # 构造两个Kd树 用于寻找领域
    knn_1 = K_NearestNeighbors(pre_matches1)
    n_dist_1, n_index_1 = knn_1.get_k_neighbors(np.array([pre_matches1[pointIndex, :]]), 16)

    knn_2 = K_NearestNeighbors(pre_matches2)
    n_dist_2, n_index_2 = knn_2.get_k_neighbors(np.array([pre_matches2[pointIndex, :]]), 16)

    # AngleSift函数的测试
    #     def __init__(self, pre_matches_1, per_matches_2, center_index_1, center_index_2,
    #                  neighbor_index_1, neighbor_index_2, neighbor_dist_1, neighbor_dist_2,
    #                   pre_matches_des_1, pre_matches_des_2):
    angle_sift = AngleSift(pre_matches1, pre_matches2, pointIndex, pointIndex, n_index_1, n_index_2,
                           n_dist_1, n_dist_2, des1, des2)
    a = angle_sift.create_sift_angle_descriptor()
    #FuzzyGlobalCircle的测试
    # 利用kd树得到全局所有点的距离
    global_n_dist_1, global_n_index_1 = knn_1.get_k_neighbors(np.array([pre_matches1[pointIndex, :]]), len1 - 1)
    global_n_dist_2, global_n_index_2 = knn_2.get_k_neighbors(np.array([pre_matches2[pointIndex, :]]), len2 - 1)
    # 放入FuzzyGlobalCircle中
    #  __init__(self, global_dist_1, global_dist_2, split):
    split = np.linspace(2, 17, 16, dtype=int)
    fuzzy_global_circle = FuzzyGlobalCircle(global_n_dist_1, global_n_dist_2, split)
    b = fuzzy_global_circle.create_fuzzy_global_circle_descriptor()
    print(b)
    c = np.hstack((a, b))

    # # # 领域的点的可视化
    # plt.figure(num='reference')
    # plt.scatter(pre_matches1_t[0, :], pre_matches1_t[1, :], s=2)
    # plt.scatter(pre_matches1_t[0, [n_index_1]], pre_matches1_t[1, [n_index_1]], c='blue', s=2)
    # plt.scatter(pre_matches1_t[0, pointIndex], pre_matches1_t[1, pointIndex], c='yellow', s=2)
    # plt.scatter([194.81637573, 201.05604553, 225.74801636], [113.3913269, 112.81473541, 154.33976746],
    #             c='red', s=2)
    #
    # plt.figure(num='sensed')
    # plt.scatter(pre_matches2_t[0, :], pre_matches2_t[1, :], s=2)
    # plt.scatter(pre_matches2_t[0, [n_index_2]], pre_matches2_t[1, [n_index_2]], c='blue', s=2)
    # plt.scatter(pre_matches2_t[0, pointIndex], pre_matches2_t[1, pointIndex], c='yellow', s=2)
    # plt.scatter([308.92025757, 314.47872925, 324.65664673], [89.91501617, 90.25621796, 126.86416626],
    #             c='red', s=2)
    #
    # plt.show()
