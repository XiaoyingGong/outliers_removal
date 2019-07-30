import numpy as np

# SIFT阈值
SIFT_THRESHOLD = 0.8

GAUSSIAN_PENALTY_SIGMA_1 = 0.6
GAUSSIAN_PENALTY_SIGMA_2 = 0.4

# 控制sift描述子的强度, 1.0
ROTATION_SCALE_INV_SIFT_WEIGHTS = 0.0


# 在原本的与预匹配点集中，抽取部分点
# PARTIAL_POINTS_K 代表 有 30份
# PARTIAL_POINTS_K_NUM 代表 每份 有 10个
PARTIAL_POINTS_K = 30
PARTIAL_POINTS_K_NUM = 10

# 在angle_sift算法中邻域的个数
ANGLE_SIFT_KNN_K = 16
# fuzzy_global_circle 每次分多少份
FUZZY_GLOBAL_CIRCLE_SPLIT = np.linspace(2, 17, 16, dtype=int)
#在intra_neighborhood_distance中的领域的长度
INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE = 16
# 在r_s_inv_des中描述子的位数,即邻域个数*2
ROTATION_SCALE_INV_KNN_K = 16

# 使用的描述子的种类
ANGLE_SIFT = "ANGLE_SIFT"
FUZZY_GLOBAL_CIRCLE = "FUZZY_GLOBAL_CIRCLE"
INTRA_NEIGHBORHOOD = "INTRA_NEIGHBORHOOD"
ROTATION_SCALE_INV = "ROTATION_SCALE_INV"

# 文件存储的位置
# 标签的位置
LABELED_DATA_PATH = "./data/labeled_data/"
# 生成的文件暂时存储的位置
TEMP_SAVE_PATH = "./data/temp_save/"
# 图片存储的位置
IMG_SAVE_PATH = "./img/"




