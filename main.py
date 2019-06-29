import matplotlib.pyplot as plt
import cv2
import numpy as np
from feature_matching import sift_matching_alter
from knn.knn import K_NearestNeighbors
# 主类，汇总各个类的功能

# 图像路径
img1_path = "./img/1.png"
img2_path = "./img/2.png"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
h_img = np.hstack((img1, img2))
#sift的阈值, 推荐设置高一点以增加负样
sift_threshold = 0.6

# 通过sift进行预匹配
pre_matches1, pre_matches2 = sift_matching_alter.get_matches(img1_path, img2_path, sift_threshold)

pointIndex = 500
pre_matches1_t = np.transpose(pre_matches1)

knn_1 = K_NearestNeighbors(pre_matches1)
v1, v2 = knn_1.get_k_neighbors(np.array([pre_matches1[pointIndex, :]]), 15, pointIndex)

pre_matches2_t = np.transpose(pre_matches2)

knn_2 = K_NearestNeighbors(pre_matches2)
v21, v22 = knn_2.get_k_neighbors(np.array([pre_matches2[pointIndex, :]]), 15, pointIndex)

plt.figure(num='reference')
plt.scatter(pre_matches1_t[0, :], pre_matches1_t[1, :], s=2)
plt.scatter(pre_matches1_t[0, [v2]],pre_matches1_t[1, [v2]], c='red', s=2)
plt.scatter(pre_matches1_t[0, pointIndex], pre_matches1_t[1, pointIndex], c='yellow', s=2)

plt.figure(num='sensed')
plt.scatter(pre_matches2_t[0, :], pre_matches2_t[1, :], s=2)
plt.scatter(pre_matches2_t[0, [v22]], pre_matches2_t[1, [v22]], c='red', s=2)
plt.scatter(pre_matches2_t[0, pointIndex], pre_matches2_t[1, pointIndex], c='yellow', s=2)

plt.show()

print(len(pre_matches1))
print(np.unique(pre_matches1, axis=0))
print(len(np.unique(pre_matches1,  axis=0)))
