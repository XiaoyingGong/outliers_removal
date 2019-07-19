import numpy as np
import cv2

from utils import constant
from feature_matching import sift_matching
from knn.knn import K_NearestNeighbors
from match_descriptor.angle_sift import AngleSift
from match_descriptor.fuzzy_global_circle import FuzzyGlobalCircle
from match_descriptor.intra_neighborhood_distance import IntraNeighborDist

# des: 用于图像提取特征点的取样, 即不要某个阈值下的所有点，只要部分点
# input: k代表分成多少分， k_num代表每份多少个
# return: pre_matches[index]代表取得k*k_num个预匹配， index代表取出来的点在原来的pre_matches中的序号
def get_partial_points(pre_matches, k, k_num):
    pre_matches_len = len(pre_matches)
    split_len = int(pre_matches_len / k)
    index = np.array([], dtype=np.int)
    split_index = np.linspace(0, k_num-1, k_num, dtype=np.int)
    for i in range(k):
        index = np.append(index, split_index)
        if i != k - 1:
            split_index += split_len
    return pre_matches[index], index


# 用于一对多的去除，强行变成一对一
def become_one_to_one(pre_matches_1, pre_matches_2):
    pre_matches_1, index_1 = np.unique(pre_matches_1, return_index=True, axis=0)
    pre_matches_2 = pre_matches_2[index_1]
    pre_matches_2, index_2 = np.unique(pre_matches_2, return_index=True, axis=0)
    pre_matches_1 = pre_matches_1[index_2]
    return pre_matches_1, pre_matches_2


# 以下为本.py文件的主函数
# k代表分成多少分， k_num代表每份多少个, is_unique代表是否需要将预匹配一一对应化，即去掉一对多的情况
# train_data： 'once':只有第一个描述子， 'twice'：只有第二个描述子，'double'：两个描述子都有
def create_descriptor(img_path_1, img_path_2, k=None, k_num=None, is_unique=True, descriptor_category='double'):
    # 图像路径
    img_1 = cv2.imread(img_path_1)
    img_2 = cv2.imread(img_path_2)
    # resize,这儿resize需要改进
    img_1 = cv2.resize(img_1, (800, 600))
    img_2 = cv2.resize(img_2, (800, 600))
    # 水平拼接后的图像
    h_img = np.hstack((img_1, img_2))
    # 通过sift进行预匹配
    pre_matches_1, pre_matches_2, des_1, des_2, match_index = \
        sift_matching.get_matches(img_1, img_2, constant.SIFT_THRESHOLD)
    # 因为匹配里面也有可能存在一对多的情况所以，这里进行一次将一对多的情况剔除
    # is_unique == true执行下列代码, is_unique == false 不执行下列代码
    # 这儿如果不这样做，用kd树求全局距离的时候（即描述子2），会出错
    # 需要改进get_k_neighbors及不排除0的情况才能弄对
    if is_unique:
        pre_matches_1, pre_matches_2 = become_one_to_one(pre_matches_1, pre_matches_2)
    # 求长度
    len_1 = len(pre_matches_1)
    len_2 = len(pre_matches_2)

    # 得到部分的点集
    # 如果k != None 或者 k_num ！= None否则是全部所有的点集一起上
    # partial_index_1 与 partial_index_2 值应该是一样的，
    # 因为 pre_matches_1 与 pre_matches_2 也是对应的
    if k != None and k_num != None:
        _, partial_index_1 = get_partial_points(pre_matches_1, k, k_num)
        _, partial_index_2 = get_partial_points(pre_matches_2, k, k_num)
    else:
        partial_index_1 = np.array(range(len(pre_matches_1)))
        partial_index_2 =  np.array(range(len(pre_matches_2)))

    if descriptor_category == 'double':
        descriptor_len = len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT) + constant.ANGLE_SIFT_KNN_K
    elif descriptor_category == 'once':
        descriptor_len = constant.ANGLE_SIFT_KNN_K
    elif descriptor_category == 'three':
        descriptor_len = constant.ANGLE_SIFT_KNN_K + len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT) + constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE
    else:
        descriptor_len = len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT)
    descriptor_final = np.zeros([len(partial_index_1), descriptor_len])
    # 产生描述子
    for i in range(len(partial_index_1)):
        pointIndex = partial_index_1[i]
        # 构造两个Kd树 用于寻找领域
        knn_1 = K_NearestNeighbors(pre_matches_1)
        knn_2 = K_NearestNeighbors(pre_matches_2)
        # 寻找邻近的点
        n_dist_1, n_index_1 = knn_1.get_k_neighbors(np.array([pre_matches_1[pointIndex, :]]), constant.ANGLE_SIFT_KNN_K)
        n_dist_2, n_index_2 = knn_2.get_k_neighbors(np.array([pre_matches_2[pointIndex, :]]), constant.ANGLE_SIFT_KNN_K)

        # 求描述子1 angle_sift
        angle_sift = AngleSift(pre_matches_1, pre_matches_2, pointIndex, pointIndex, n_index_1, n_index_2,
                               n_dist_1, n_dist_2, des_1, des_2)
        descriptor_a = angle_sift.create_sift_angle_descriptor()

        # 求描述子2 fuzzy_global_circle
        # 利用kd树求该点到全局个点的距离
        global_n_dist_1, global_n_index_1 = knn_1.get_k_neighbors(np.array([pre_matches_1[pointIndex, :]]), len_1 - 1)
        global_n_dist_2, global_n_index_2 = knn_2.get_k_neighbors(np.array([pre_matches_2[pointIndex, :]]), len_2 - 1)
        # fuzzy_global_circle 每次分多少份
        split = constant.FUZZY_GLOBAL_CIRCLE_SPLIT
        fuzzy_global_circle = FuzzyGlobalCircle(global_n_dist_1, global_n_dist_2, split)
        descriptor_b, _ = fuzzy_global_circle.create_fuzzy_global_circle_descriptor()
        # IntraNeighborDist
        intra_neighbor_dist = IntraNeighborDist(n_dist_1, n_dist_2)
        descriptor_c = intra_neighbor_dist.cal_intra_neighbor_dist_descriptor()

        # 将两个描述子拼在一起成为一个向量
        if descriptor_category == 'double':
            descriptor_final[i] = np.hstack((descriptor_a, descriptor_b))
        elif descriptor_category == 'once':
            descriptor_final[i] = descriptor_a
        elif descriptor_category == 'three':
            descriptor_final[i] = np.hstack((descriptor_a, descriptor_b, descriptor_c))
        else:
            descriptor_final[i] = descriptor_b

    return descriptor_final, pre_matches_1, pre_matches_2, h_img


