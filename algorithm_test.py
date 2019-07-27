import numpy as np
import matplotlib.pyplot as plt
from utils import constant, utils
from network.outliers_removal_network_48bit_softmax import ORNet
from descriptor_operation import create_descriptor, create_pre_matches
import cv2
constant.SIFT_THRESHOLD = 1.0
# 读图
# 性能瓶颈：constant.FUZZY_GLOBAL_CIRCLE 可以从编程上改善
img_path_1 = "./img/15_r.jpg"
img_path_2 = "./img/15_s.jpg"
# my_descriptor = np.zeros([100, 48])

pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, img_1, img_2 = create_pre_matches.get_pre_matches(img_path_1, img_path_2, is_unique=True, k=None, k_num=None)
for i in range(len(des_1)):
    print(utils.euclidean_distance(des_1[i], des_2[i]))
    print(1 - (1 / np.sqrt(2) * utils.standard_euclidean_distance_standard_vector(des_1[i], des_2[i])))

