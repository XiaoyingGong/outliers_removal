import numpy as np
import matplotlib.pyplot as plt
from feature_matching import sift_matching
from knn.knn import K_NearestNeighbors
from match_descriptor.angle_sift import AngleSift
from match_descriptor.fuzzy_global_circle import FuzzyGlobalCircle
from network.outliers_removal_network import ORNet
from descriptor_operation import create_descriptor
import cv2
# 读图
img_path_1 = "./img/15_r.jpg"
img_path_2 = "./img/15_s.jpg"
my_descriptor = np.zeros([100, 32])
descriptoer, pre_matches_1, pre_matches_2, h_img = create_descriptor.create_descriptor(img_path_1, img_path_2, k=None, k_num=None, is_unique=True, descriptor_category='double')
pre_matches_1 = pre_matches_1
pre_matches_2 = pre_matches_2
my_descriptor = descriptoer
print(len(my_descriptor))
# 将prematch转置，便于matplotlib绘制
pre_matches1_t = np.transpose(pre_matches_1)
pre_matches2_t = np.transpose(pre_matches_2)

# 声明网络
ornet = ORNet("./model/48bit/model")
predict = ornet.predict(my_descriptor)

inlier_index = np.where(np.argmax(predict, axis=1) == 0)[0]
print(len(inlier_index))
points_1 = np.transpose(pre_matches_1)
points_2 = np.transpose(pre_matches_2)

plt.figure(num="未去除误匹配")
plt.scatter(points_1[0, :], points_1[1, :], s=2, c="red")
plt.scatter(points_2[0, :] + 800, points_2[1, :], s=2, c="red")
plt.plot([points_1[0, :], points_2[0, :] + 800], [points_1[1, :], points_2[1, :]], linewidth=1, c="yellow")
plt.imshow(h_img)

points_1 = points_1[:, inlier_index]
points_2 = points_2[:, inlier_index]
plt.figure(num="去除误匹配")
plt.scatter(points_1[0, :], points_1[1, :], s=2, c="red")
plt.scatter(points_2[0, :] + 800, points_2[1, :], s=2, c="red")
plt.plot([points_1[0, :], points_2[0, :] + 800], [points_1[1, :], points_2[1, :]], linewidth=1, c="yellow")
plt.imshow(h_img)

plt.show()
