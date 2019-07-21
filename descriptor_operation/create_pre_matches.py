from utils import constant
from feature_matching import sift_matching
import cv2
import numpy as np

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


def get_pre_matches(img_path_1, img_path_2, is_unique=True, k=None, k_num=None):
    # 图像路径
    img_1 = cv2.imread(img_path_1)
    img_2 = cv2.imread(img_path_2)
    # resize,这儿resize需要改进
    img_1 = cv2.resize(img_1, (800, 600))
    img_2 = cv2.resize(img_2, (800, 600))

    # 通过sift进行预匹配
    pre_matches_1, pre_matches_2, des_1, des_2, match_index = \
        sift_matching.get_matches(img_1, img_2, constant.SIFT_THRESHOLD)
    # 因为匹配里面也有可能存在一对多的情况所以，这里进行一次将一对多的情况剔除
    # is_unique == true执行下列代码, is_unique == false 不执行下列代码
    # 这儿如果不这样做，用kd树求全局距离的时候（即描述子2），会出错
    # 需要改进get_k_neighbors及不排除0的情况才能弄对
    if is_unique:
        pre_matches_1, pre_matches_2 = become_one_to_one(pre_matches_1, pre_matches_2)

    # 得到部分的点集
    # 如果k != None 或者 k_num ！= None否则是全部所有的点集一起上
    # partial_index_1 与 partial_index_2 值应该是一样的，
    # 因为 pre_matches_1 与 pre_matches_2 也是对应的
    if k is not None and k_num is not None:
        _, partial_index_1 = get_partial_points(pre_matches_1, k, k_num)
        _, partial_index_2 = get_partial_points(pre_matches_2, k, k_num)
    else:
        partial_index_1 = np.array(range(len(pre_matches_1)))
        partial_index_2 = np.array(range(len(pre_matches_2)))

    return pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, img_1, img_2
