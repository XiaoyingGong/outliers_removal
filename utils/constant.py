import numpy as np

GAUSSIAN_PENALTY_SIGMA_1 = 0.6

GAUSSIAN_PENALTY_SIGMA_2 = 1.2

# 在原本的与预匹配点集中，抽取部分点
# PARTIAL_POINTS_K 代表 有 30份
# PARTIAL_POINTS_K_NUM 代表 每份 有 10个
PARTIAL_POINTS_K = 30
PARTIAL_POINTS_K_NUM = 10

# SIFT阈值
SIFT_THRESHOLD = 0.85
# 在angle_sift算法中邻域的个数
ANGLE_SIFT_KNN_K = 16
# fuzzy_global_circle 每次分多少份
FUZZY_GLOBAL_CIRCLE_SPLIT = np.linspace(2, 17, 16, dtype=int)
#在intra_neighborhood_distance中的领域的长度
INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE = 16





