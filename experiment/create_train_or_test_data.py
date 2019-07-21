import numpy as np

from utils import constant
from descriptor_operation import create_descriptor
from descriptor_operation import create_pre_matches


def create_train_or_test_data(img_r_list, img_s_list, labeled_data_list, search_index, train_or_test="train", descriptor_categories=None):
    labeled_data_path = constant.LABELED_DATA_PATH
    save_path = constant.TEMP_SAVE_PATH
    train_or_test = train_or_test

    des_len = create_descriptor.get_descriptor_len(descriptor_categories)
    # 每张图片提取多少个点
    K_MUL_K_NUM = constant.PARTIAL_POINTS_K * constant.PARTIAL_POINTS_K_NUM
    # 最后要输出的Descriptoers和Labels为一个矩阵，所以大写区分
    Descriptors = np.zeros([len(img_r_list[search_index]) * K_MUL_K_NUM, des_len])
    # [1, 0]为inlier即1  [0, 1]为outlier即0
    Labels = np.zeros([len(img_r_list[search_index]) * K_MUL_K_NUM, 2])
    Descriptors_index_start = 0
    Labels_i = 0

    for img_index in search_index:
        print("正在处理序列中的第", img_index, "张图片")
        img_path_1 = constant.IMG_SAVE_PATH + img_r_list[img_index]
        img_path_2 = constant.IMG_SAVE_PATH + img_s_list[img_index]
        pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, _, _ =\
            create_pre_matches.get_pre_matches(img_path_1, img_path_2, is_unique=True, k=constant.PARTIAL_POINTS_K, k_num=constant.PARTIAL_POINTS_K_NUM)
        # 产生描述子
        descriptor_final = create_descriptor.create_descriptor(pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, descriptor_categories=descriptor_categories)
        Descriptors[Descriptors_index_start: Descriptors_index_start + K_MUL_K_NUM] = descriptor_final
        Descriptors_index_start += K_MUL_K_NUM
        # 读取标注
        load_labeled = np.load(labeled_data_path + labeled_data_list[img_index])
        the_labels = load_labeled["correspondence_label"][4]
        for the_label in the_labels:
            if the_label == 1.0:
                Labels[Labels_i] = [1, 0]
            else:
                Labels[Labels_i] = [0, 1]
            Labels_i += 1
    #保存
    if train_or_test == "train":
        np.savez(save_path + "train_data", train_descriptor=Descriptors, train_label=Labels)
    else:
        np.savez(save_path + "test_data", test_descriptor=Descriptors, test_label=Labels)
