import numpy as np
import cv2

from utils import constant
from feature_matching import sift_matching
from knn.knn import K_NearestNeighbors
from match_descriptor.angle_sift import AngleSift
from match_descriptor.fuzzy_global_circle import FuzzyGlobalCircle
from match_descriptor.intra_neighborhood_distance import IntraNeighborDist
from match_descriptor.r_s_inv_des import RotationScaleInvDes
from match_descriptor.r_s_inv_des_inlier import RotationScaleInvDes as RotationScaleInvDesInlier

#  input: 根据kd树得到的 距离 和该距离的点的 下标, 当前产生描述子的点下标point_index, 预匹配_1, 预匹配_2
#  根据constant的配置，得到angle_sift描述子
def get_angle_sift_des(n_dist_1, n_dist_2, n_index_1, n_index_2, point_index, pre_matches_1,
                       pre_matches_2, des_1, des_2):
    angle_sift = AngleSift(pre_matches_1, pre_matches_2, point_index, point_index, n_index_1, n_index_2,
                           n_dist_1, n_dist_2, des_1, des_2)
    angle_sift_des = angle_sift.create_sift_angle_descriptor()
    return angle_sift_des


# input: global_n_dist_1, global_n_dist_2：根据kd树得到的全局距离。
# 根据constant的配置，得到fuzzy_global_circle描述子：
def get_fuzzy_global_circle_des(global_n_dist_1, global_n_dist_2):
    split = constant.FUZZY_GLOBAL_CIRCLE_SPLIT
    fuzzy_global_circle = FuzzyGlobalCircle(global_n_dist_1, global_n_dist_2, split)
    fuzzy_global_circle_des, _ = fuzzy_global_circle.create_fuzzy_global_circle_descriptor()
    return fuzzy_global_circle_des


#  input: n_dist_1，n_dist_2， 根据kd树得到的该点的领域距离
# 根据constant的配置，得到intra_neighbor_dist
def get_intra_neighbor_dist_des(n_dist_1, n_dist_2):
    intra_neighbor_dist = IntraNeighborDist(n_dist_1, n_dist_2)
    intra_neighbor_dist = intra_neighbor_dist.get_intra_neighbor_dist_descriptor()
    return intra_neighbor_dist

# 描述子4
def get_r_s_inv_des(n_dist_1, n_dist_2, n_index_1, n_index_2, point_index, pre_matches_1,
                       pre_matches_2, des_1, des_2):
    r_s_inv = RotationScaleInvDes(pre_matches_1, pre_matches_2, des_1, des_2, point_index, point_index, n_index_1, n_index_2,
                        n_dist_1, n_dist_2)
    descriptor = r_s_inv.create_r_s_inv_descriptor()
    return descriptor

# 描述子4 改进版本
def get_r_s_inv_des_inlier(n_dist_1, n_dist_2, n_index_1, n_index_2, point_index, pre_matches_1,
                       pre_matches_2, des_1, des_2, inlier_index):
    r_s_inv = RotationScaleInvDesInlier(pre_matches_1, pre_matches_2, des_1, des_2, point_index, point_index, n_index_1, n_index_2,
                        n_dist_1, n_dist_2)
    descriptor = r_s_inv.create_r_s_inv_descriptor(inlier_index)
    return descriptor

# 根据输入的需要的“描述子类别”向量
# 获得产生的描述子的长度
def get_descriptor_len(descriptor_categories):
    des_cate_len = len(descriptor_categories)
    descriptor_len = 0
    for i in range(des_cate_len):
        # 描述子1：角度与SIFT
        if descriptor_categories[i] == constant.ANGLE_SIFT:
            descriptor_len += constant.ANGLE_SIFT_KNN_K
        # 描述子2： 模糊的全局同心圆计数
        elif descriptor_categories[i] == constant.FUZZY_GLOBAL_CIRCLE:
            descriptor_len += len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT)
        # 描述子3： 领域距离的比值
        elif descriptor_categories[i] == constant.INTRA_NEIGHBORHOOD:
            descriptor_len += constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE
        # 描述子4:旋转和尺度不变的描述子
        else:
            descriptor_len += constant.ROTATION_SCALE_INV_KNN_K * 2
    return descriptor_len


# 以下为本.py文件的主函数
# k代表分成多少分， k_num代表每份多少个, is_unique代表是否需要将预匹配一一对应化，即去掉一对多的情况
# train_data： 'once':只有第一个描述子， 'twice'：只有第二个描述子，'double'：两个描述子都有
def create_descriptor(pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, descriptor_categories=None):
    # 最后输出的描述子
    descriptor_len = get_descriptor_len(descriptor_categories)
    descriptor_final = np.zeros([len(partial_index_1), descriptor_len])
    # 构造两个Kd树 用于寻找领域
    knn_1 = K_NearestNeighbors(pre_matches_1)
    knn_2 = K_NearestNeighbors(pre_matches_2)
    # 两个pre_match的长度
    len_1 = len(pre_matches_1)
    len_2 = len(pre_matches_2)
    print("KNN is ok")
    # 产生描述子
    for i in range(len(partial_index_1)):
        point_index = partial_index_1[i]

        # 寻找邻近的点 angle_sift 和 Intra_Neighborhood_dist这两个算子都可能用到他们
        if (descriptor_categories == constant.ANGLE_SIFT).any() or \
                (descriptor_categories == constant.INTRA_NEIGHBORHOOD).any():
            n_dist_1, n_index_1 =\
                knn_1.get_k_neighbors(np.array([pre_matches_1[point_index, :]]), constant.ANGLE_SIFT_KNN_K)
            n_dist_2, n_index_2 =\
                knn_2.get_k_neighbors(np.array([pre_matches_2[point_index, :]]), constant.ANGLE_SIFT_KNN_K)

        # scale and rotation inv的描述子
        if (descriptor_categories == constant.ROTATION_SCALE_INV).any():
            n_dist_inv_1, n_index_inv_1 =\
                knn_1.get_k_neighbors(np.array([pre_matches_1[point_index, :]]), constant.ROTATION_SCALE_INV_KNN_K)
            n_dist_inv_2, n_index_inv_2 = \
                knn_2.get_k_neighbors(np.array([pre_matches_2[point_index, :]]), constant.ROTATION_SCALE_INV_KNN_K)

        # 求描述子1：angle_sift
        if (descriptor_categories == constant.ANGLE_SIFT).any():
            angle_sift_des = get_angle_sift_des(n_dist_1, n_dist_2, n_index_1, n_index_2, point_index, pre_matches_1,
                           pre_matches_2, des_1, des_2)

        # 利用kd树求该点到全局个点的距离
        if (descriptor_categories == constant.FUZZY_GLOBAL_CIRCLE).any():
            global_n_dist_1, global_n_index_1 =\
                knn_1.get_k_neighbors(np.array([pre_matches_1[point_index, :]]), len_1 - 1)
            global_n_dist_2, global_n_index_2 =\
                knn_2.get_k_neighbors(np.array([pre_matches_2[point_index, :]]), len_2 - 1)
            # 求描述子2：fuzzy_global_circle
            fuzzy_global_circle_des = get_fuzzy_global_circle_des(global_n_dist_1, global_n_dist_2)

        # 求描述子3：IntraNeighborDist
        if (descriptor_categories == constant.INTRA_NEIGHBORHOOD).any():
            intra_neighbor_dist_des = get_intra_neighbor_dist_des(n_dist_1, n_dist_2)

        # 求描述子4:
        if (descriptor_categories == constant.ROTATION_SCALE_INV).any():
            r_s_inv_des = get_r_s_inv_des(n_dist_inv_1, n_dist_inv_2, n_index_inv_1, n_index_inv_2, point_index,
                                          pre_matches_1, pre_matches_2, des_1, des_2)


        des_cate_i = 0
        for descriptor_category in descriptor_categories:
            if descriptor_category == constant.ANGLE_SIFT:
                descriptor_final[i, des_cate_i:des_cate_i+constant.ANGLE_SIFT_KNN_K] =\
                    angle_sift_des
                des_cate_i += constant.ANGLE_SIFT_KNN_K
            elif descriptor_category == constant.FUZZY_GLOBAL_CIRCLE:
                descriptor_final[i, des_cate_i: des_cate_i + len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT)] =\
                    fuzzy_global_circle_des
                des_cate_i += len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT)
            elif descriptor_category == constant.INTRA_NEIGHBORHOOD:
                descriptor_final[i, des_cate_i: des_cate_i + constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE] =\
                    intra_neighbor_dist_des
                des_cate_i += constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE
            else:
                descriptor_final[i, des_cate_i: des_cate_i + constant.ROTATION_SCALE_INV_KNN_K * 2] =\
                    r_s_inv_des
                des_cate_i += constant.ROTATION_SCALE_INV_KNN_K * 2
    return descriptor_final


# 以下为本.py文件的主函数
# k代表分成多少分， k_num代表每份多少个, is_unique代表是否需要将预匹配一一对应化，即去掉一对多的情况
# train_data： 'once':只有第一个描述子， 'twice'：只有第二个描述子，'double'：两个描述子都有
def create_descriptor_inlier_version(pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, inlier_index, descriptor_categories=None):
    # 最后输出的描述子
    descriptor_len = get_descriptor_len(descriptor_categories)
    descriptor_final = np.zeros([len(partial_index_1), descriptor_len])
    # 构造两个Kd树 用于寻找领域
    knn_1 = K_NearestNeighbors(pre_matches_1)
    knn_2 = K_NearestNeighbors(pre_matches_2)
    # 两个pre_match的长度
    len_1 = len(pre_matches_1)
    len_2 = len(pre_matches_2)
    print("KNN is ok")
    # 产生描述子
    for i in range(len(partial_index_1)):
        point_index = partial_index_1[i]

        # 寻找邻近的点 angle_sift 和 Intra_Neighborhood_dist这两个算子都可能用到他们
        if (descriptor_categories == constant.ANGLE_SIFT).any() or \
                (descriptor_categories == constant.INTRA_NEIGHBORHOOD).any():
            n_dist_1, n_index_1 =\
                knn_1.get_k_neighbors(np.array([pre_matches_1[point_index, :]]), constant.ANGLE_SIFT_KNN_K)
            n_dist_2, n_index_2 =\
                knn_2.get_k_neighbors(np.array([pre_matches_2[point_index, :]]), constant.ANGLE_SIFT_KNN_K)

        # scale and rotation inv的描述子
        if (descriptor_categories == constant.ROTATION_SCALE_INV).any():
            n_dist_inv_1, n_index_inv_1 =\
                knn_1.get_k_neighbors(np.array([pre_matches_1[point_index, :]]), constant.ROTATION_SCALE_INV_KNN_K)
            n_dist_inv_2, n_index_inv_2 = \
                knn_2.get_k_neighbors(np.array([pre_matches_2[point_index, :]]), constant.ROTATION_SCALE_INV_KNN_K)

        # 求描述子1：angle_sift
        if (descriptor_categories == constant.ANGLE_SIFT).any():
            angle_sift_des = get_angle_sift_des(n_dist_1, n_dist_2, n_index_1, n_index_2, point_index, pre_matches_1,
                           pre_matches_2, des_1, des_2)

        # 利用kd树求该点到全局个点的距离
        if (descriptor_categories == constant.FUZZY_GLOBAL_CIRCLE).any():
            global_n_dist_1, global_n_index_1 =\
                knn_1.get_k_neighbors(np.array([pre_matches_1[point_index, :]]), len_1 - 1)
            global_n_dist_2, global_n_index_2 =\
                knn_2.get_k_neighbors(np.array([pre_matches_2[point_index, :]]), len_2 - 1)
            # 求描述子2：fuzzy_global_circle
            fuzzy_global_circle_des = get_fuzzy_global_circle_des(global_n_dist_1, global_n_dist_2)

        # 求描述子3：IntraNeighborDist
        if (descriptor_categories == constant.INTRA_NEIGHBORHOOD).any():
            intra_neighbor_dist_des = get_intra_neighbor_dist_des(n_dist_1, n_dist_2)

        # 求描述子4:
        if (descriptor_categories == constant.ROTATION_SCALE_INV).any():
            r_s_inv_des = get_r_s_inv_des_inlier(n_dist_inv_1, n_dist_inv_2, n_index_inv_1, n_index_inv_2, point_index,
                                          pre_matches_1, pre_matches_2, des_1, des_2, inlier_index)


        des_cate_i = 0
        for descriptor_category in descriptor_categories:
            if descriptor_category == constant.ANGLE_SIFT:
                descriptor_final[i, des_cate_i:des_cate_i+constant.ANGLE_SIFT_KNN_K] =\
                    angle_sift_des
                des_cate_i += constant.ANGLE_SIFT_KNN_K
            elif descriptor_category == constant.FUZZY_GLOBAL_CIRCLE:
                descriptor_final[i, des_cate_i: des_cate_i + len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT)] =\
                    fuzzy_global_circle_des
                des_cate_i += len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT)
            elif descriptor_category == constant.INTRA_NEIGHBORHOOD:
                descriptor_final[i, des_cate_i: des_cate_i + constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE] =\
                    intra_neighbor_dist_des
                des_cate_i += constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE
            else:
                descriptor_final[i, des_cate_i: des_cate_i + constant.ROTATION_SCALE_INV_KNN_K * 2] =\
                    r_s_inv_des
                des_cate_i += constant.ROTATION_SCALE_INV_KNN_K * 2
    return descriptor_final
