import numpy as np
import matplotlib.pyplot as plt
from utils import constant
from network.outliers_removal_network import ORNet
from descriptor_operation import create_descriptor, create_pre_matches
import cv2
# 读图
# 性能瓶颈：constant.FUZZY_GLOBAL_CIRCLE 可以从编程上改善
img_path_1 = "./img/16_r.png"
img_path_2 = "./img/16_s.png"
constant.SIFT_THRESHOLD = 0.9
# my_descriptor = np.zeros([100, 48])
descriptor_categories = np.array([constant.ROTATION_SCALE_INV])
pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, img_1, img_2, resize_h, resize_w = create_pre_matches.get_pre_matches(img_path_1, img_path_2, is_unique=True, k=None, k_num=None)
descriptor_final = create_descriptor.create_descriptor(
    pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, descriptor_categories=descriptor_categories)

print(img_1.shape)
my_descriptor = descriptor_final

h_img = np.hstack((img_1, img_2))
print(len(my_descriptor))
# 将prematch转置，便于matplotlib绘制
pre_matches1_t = np.transpose(pre_matches_1)
pre_matches2_t = np.transpose(pre_matches_2)

# 声明网络
ornet = ORNet("./model/32bit/model")
predict = ornet.predict(my_descriptor)

inlier_index = np.where(np.argmax(predict, axis=1) == 0)[0]
print(len(inlier_index))
points_1 = np.transpose(pre_matches_1)
points_2 = np.transpose(pre_matches_2)

plt.figure(num="未去除误匹配")
plt.scatter(points_1[0, :], points_1[1, :], s=2, c="red")
plt.scatter(points_2[0, :] + resize_w, points_2[1, :], s=2, c="red")
plt.plot([points_1[0, :], points_2[0, :] + resize_w], [points_1[1, :], points_2[1, :]], linewidth=1, c="yellow")
plt.imshow(h_img)

points_1 = points_1[:, inlier_index]
points_2 = points_2[:, inlier_index]
plt.figure(num="去除误匹配")
plt.scatter(points_1[0, :], points_1[1, :], s=2, c="red")
plt.scatter(points_2[0, :] + resize_w, points_2[1, :], s=2, c="red")
plt.plot([points_1[0, :], points_2[0, :] + resize_w], [points_1[1, :], points_2[1, :]], linewidth=1, c="yellow")
plt.imshow(h_img)

plt.show()
