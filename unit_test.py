from descriptor_operation import create_pre_matches
from match_descriptor.r_s_inv_des import RotationScaleInvDes
from knn.knn import K_NearestNeighbors
import numpy as np
from utils import utils, constant

img_path_1 = "./img/1_r.png"
img_path_2 = "./img/1_s.png"
is_unique = True
k = None
k_num = None

pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, img_1, img_2 = \
    create_pre_matches.get_pre_matches(img_path_1, img_path_2, is_unique, k, k_num)


# 构造两个Kd树 用于寻找领域
knn_1 = K_NearestNeighbors(pre_matches_1)
knn_2 = K_NearestNeighbors(pre_matches_2)

for i in range(1):
    point_index = i
    n_dist_1, n_index_1 = \
        knn_1.get_k_neighbors(np.array([pre_matches_1[point_index, :]]), constant.ROTATION_SCALE_INV_KNN_K)
    n_dist_2, n_index_2 = \
        knn_2.get_k_neighbors(np.array([pre_matches_2[point_index, :]]), constant.ROTATION_SCALE_INV_KNN_K)
    r_s_i_d = RotationScaleInvDes(pre_matches_1, pre_matches_2, des_1, des_2, i, i, n_index_1, n_index_2, n_dist_1, n_dist_2)
    descriptor = r_s_i_d.create_r_s_inv_descriptor()
    print(descriptor)
