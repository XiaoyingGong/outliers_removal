import numpy as np

from utils import constant
from descriptor_operation.create_descriptor import create_descriptor


def create_train_or_test_data(img_r_list, img_s_list, labeled_data_list, search_index, train_or_test="train", descriptor_category="double"):
    labeled_data_path = "./data/labeled_data/"
    save_path = "./data/temp_save/"
    train_or_test = train_or_test
    if descriptor_category == "double":
        train_descriptor_len = constant.ANGLE_SIFT_KNN_K + len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT)
    elif descriptor_category == "once":
        train_descriptor_len = constant.ANGLE_SIFT_KNN_K
    elif descriptor_category == "three":
        train_descriptor_len = constant.ANGLE_SIFT_KNN_K + len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT) + constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE
    else:
        train_descriptor_len = len(constant.FUZZY_GLOBAL_CIRCLE_SPLIT)

    train_descriptor = np.zeros([3300, 48])
    # [1, 0]为inlier即1  [0, 1]为outlier即0
    train_label = np.zeros([3300, 2])
    index_start = 0
    index_end = 300
    train_label_i = 0
    for img_index in search_index:
        print("正在处理序列中的第", img_index, "张图片")
        img_path_1 = "./img/" + img_r_list[img_index]
        img_path_2 = "./img/" + img_s_list[img_index]

        print("index_start:", index_start, "index_end:", index_end)
        # 产生描述子
        descriptor, _, _, _ = create_descriptor(img_path_1, img_path_2, k=constant.PARTIAL_POINTS_K,
                                       k_num=constant.PARTIAL_POINTS_K_NUM, is_unique=True, descriptor_category=descriptor_category)
        train_descriptor[index_start: index_end] = descriptor
        index_start = index_start + 300
        index_end = index_end + 300
        # 读取标注
        load_labeled = np.load(labeled_data_path + labeled_data_list[img_index])
        the_labels = load_labeled["correspondence_label"][4]
        for the_label in the_labels:
            if the_label == 1.0:
                train_label[train_label_i] = [1, 0]
            else:
                train_label[train_label_i] = [0, 1]
            train_label_i += 1

    #保存
    if train_or_test == "train":
        np.savez(save_path+ "train_data", train_descriptor=train_descriptor, train_label=train_label)
    else:
        np.savez(save_path + "test_data", test_descriptor=train_descriptor, test_label=train_label)
