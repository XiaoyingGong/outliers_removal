import matplotlib.pyplot as plt
import cv2
import numpy as np
from feature_matching import sift_matching
from knn.knn import K_NearestNeighbors
from match_descriptor.angle_sift import AngleSift
from match_descriptor.fuzzy_global_circle import FuzzyGlobalCircle




# 这个类用于部分的标记 即一张图提了点 只标记300个点
def get_partial_points(pre_matches):
    pre_matches_len = len(pre_matches)
    split_len = int(pre_matches_len / 30)
    index = np.array([], dtype=np.int)
    split_index = np.linspace(0, 9, 10, dtype=np.int)
    for i in range(30):
        index = np.append(index, split_index)
        if i != 29:
            split_index += split_len
    return pre_matches[index], index


img_r_list = np.array([
    "1_r.png", "2_r.png", "3_r.png", "4_r.png",
    "5_r.png", "6_r.png", "7_r.png", "8_r.png",
    "9_r.jpg", "10_r.jpg", "11_r.jpg", "12_r.jpg"
])
img_s_list = np.array([
    "1_s.png", "2_s.png", "3_s.png", "4_s.png",
    "5_s.png", "6_s.png", "7_s.png", "8_s.png",
    "9_s.jpg", "10_s.jpg", "11_s.jpg", "12_s.jpg"
])
labeled_data_list = np.array(["1_r.png_1_s.png_1.0.npz", "2_r.png_2_s.png_1.0.npz", "3_r.png_3_s.png_1.0.npz",
                           "4_r.png_4_s.png_1.0.npz", "5_r.png_5_s.png_1.0.npz", "6_r.png_6_s.png_1.0.npz",
                           "7_r.png_7_s.png_1.0.npz", "8_r.png_8_s.png_1.0.npz", "9_r.jpg_9_s.jpg_1.0.npz",
                           "10_r.jpg_10_s.jpg_1.0.npz", "11_r.jpg_11_s.jpg_1.0.npz", "12_r.jpg_12_s.jpg_1.0.npz"])
labeled_data_path = "./data/labeled_data/"
train_data_path = "./data/train_data/"
# 定义
train_descriptor = np.zeros([300, 16])
# [1, 0]为inlier即1  [0, 1]为outlier即0
train_label = np.zeros([300, 2])
train_descriptor_i = 0
train_label_i = 0

for img_index in [0]:
    print("train_label_i", train_label_i)
    print("train_descriptor_i", train_descriptor_i)
    print("img_index", img_index)
    # 图像路径
    img1_path = "./img/" + img_r_list[img_index]
    img2_path = "./img/" + img_s_list[img_index]
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    # resize
    img1 = cv2.resize(img1, (800, 600))
    img2 = cv2.resize(img2, (800, 600))
    h_img = np.hstack((img1, img2))
    sift_threshold = 1.0

    # 通过sift进行预匹配
    pre_matches1, pre_matches2, des1, des2, match_index = sift_matching.get_matches(img1, img2, sift_threshold)
    # 因为匹配里面也有可能存在一对多的情况所以，这里进行一次将一对多的情况剔除
    pre_matches1, index1 = np.unique(pre_matches1, return_index=True, axis=0)
    pre_matches2 = pre_matches2[index1]
    pre_matches2, index2 = np.unique(pre_matches2, return_index=True, axis=0)
    pre_matches1 = pre_matches1[index2]

    len1 = len(pre_matches1)
    len2 = len(pre_matches2)

    _, partial_index1 = get_partial_points(pre_matches1)
    _, partial_index2 = get_partial_points(pre_matches2)
    # 产生描述子
    for i in range(300):
        pointIndex = partial_index1[i]# 69
        # 将prematch转置，便于matplotlib绘制
        pre_matches1_t = np.transpose(pre_matches1)
        pre_matches2_t = np.transpose(pre_matches2)

        # 构造两个Kd树 用于寻找领域
        knn_1 = K_NearestNeighbors(pre_matches1)
        n_dist_1, n_index_1 = knn_1.get_k_neighbors(np.array([pre_matches1[pointIndex, :]]), 16)

        knn_2 = K_NearestNeighbors(pre_matches2)
        n_dist_2, n_index_2 = knn_2.get_k_neighbors(np.array([pre_matches2[pointIndex, :]]), 16)

        # AngleSift函数的测试
        #     def __init__(self, pre_matches_1, per_matches_2, center_index_1, center_index_2,
        #                  neighbor_index_1, neighbor_index_2, neighbor_dist_1, neighbor_dist_2,
        #                   pre_matches_des_1, pre_matches_des_2):
        angle_sift = AngleSift(pre_matches1, pre_matches2, pointIndex, pointIndex, n_index_1, n_index_2,
                               n_dist_1, n_dist_2, des1, des2)
        a = angle_sift.create_sift_angle_descriptor()
        #FuzzyGlobalCircle的测试
        # 利用kd树得到全局所有点的距离
        global_n_dist_1, global_n_index_1 = knn_1.get_k_neighbors(np.array([pre_matches1[pointIndex, :]]), len1 - 1)
        global_n_dist_2, global_n_index_2 = knn_2.get_k_neighbors(np.array([pre_matches2[pointIndex, :]]), len2 - 1)
        # 放入FuzzyGlobalCircle中
        #  __init__(self, global_dist_1, global_dist_2, split):
        split = np.linspace(2, 17, 16, dtype=int)
        fuzzy_global_circle = FuzzyGlobalCircle(global_n_dist_1, global_n_dist_2, split)
        b, points_index = fuzzy_global_circle.create_fuzzy_global_circle_descriptor()
        c = np.hstack((a, b))
        train_descriptor[train_descriptor_i] = a
        train_descriptor_i += 1

    # 读取标注
    load_labeled = np.load(labeled_data_path + labeled_data_list[img_index])
    the_labels = load_labeled["correspondence_label"][4]
    for the_label in the_labels:
        if the_label == 1.0:
            train_label[train_label_i] = [1, 0]
        else:
            train_label[train_label_i] = [0, 1]
        train_label_i += 1


np.savez(train_data_path+"test_data", test_descriptor=train_descriptor, test_label=train_label)