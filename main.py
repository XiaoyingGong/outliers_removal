import matplotlib.pyplot as plt
import cv2
import numpy as np
from feature_matching import sift_matching
from knn.knn import K_NearestNeighbors
from match_descriptor.angle_sift import AngleSift
# 主类，汇总各个类的功能

# 图像路径
img1_path = "./img/1.png"
img2_path = "./img/2.png"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
h_img = np.hstack((img1, img2))

sift_threshold = 0.6

# 通过sift进行预匹
# 配
pre_matches1, pre_matches2, des1, des2, match_index = sift_matching.get_matches(img1_path, img2_path, sift_threshold)

for i in np.linspace(101, 600, 500, dtype=int):
    pointIndex = i # 69
    # 将prematch转置，便于matplotlib绘制
    pre_matches1_t = np.transpose(pre_matches1)
    pre_matches2_t = np.transpose(pre_matches2)

    # 构造两个Kd树 用于寻找领域
    knn_1 = K_NearestNeighbors(pre_matches1)
    n_dist_1, n_index_1 = knn_1.get_k_neighbors(np.array([pre_matches1[pointIndex, :]]), 16)

    knn_2 = K_NearestNeighbors(pre_matches2)
    n_dist_2, n_index_2 = knn_2.get_k_neighbors(np.array([pre_matches2[pointIndex, :]]), 16)

    # # # 领域的点的可视化
    # plt.figure(num='reference')
    # plt.scatter(pre_matches1_t[0, :], pre_matches1_t[1, :], s=2)
    # plt.scatter(pre_matches1_t[0, [n_index_1]], pre_matches1_t[1, [n_index_1]], c='red', s=2)
    # plt.scatter(pre_matches1_t[0, pointIndex], pre_matches1_t[1, pointIndex], c='yellow', s=2)
    #
    # plt.figure(num='sensed')
    # plt.scatter(pre_matches2_t[0, :], pre_matches2_t[1, :], s=2)
    # plt.scatter(pre_matches2_t[0, [n_index_2]], pre_matches2_t[1, [n_index_2]], c='red', s=2)
    # plt.scatter(pre_matches2_t[0, pointIndex], pre_matches2_t[1, pointIndex], c='yellow', s=2)
    #
    # plt.show()

    # AngleSift函数的测试
    #     def __init__(self, pre_matches_1, per_matches_2, center_index_1, center_index_2,
    #                  neighbor_index_1, neighbor_index_2, neighbor_dist_1, neighbor_dist_2):
    angle_sift = AngleSift(pre_matches1, pre_matches2, pointIndex, pointIndex, n_index_1, n_index_2,
                           n_dist_1, n_dist_2)
    b = angle_sift.create_sift_angle_descriptor()
    print(b)
