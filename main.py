import numpy as np

from experiment import create_train_or_test_data
from utils import constant

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

descriptor_categories = np.array([constant.INTRA_NEIGHBORHOOD])
# descriptor_categories = np.array([constant.ANGLE_SIFT, constant.FUZZY_GLOBAL_CIRCLE, constant.INTRA_NEIGHBORHOOD])
# train_or_test = "test"
train_or_test = "train"
# search_index = np.array([0])
search_index = np.array([1,2,3,4,5,6,7,8,9,10,11])
create_train_or_test_data.create_train_or_test_data(img_r_list, img_s_list, labeled_data_list, search_index,  train_or_test, descriptor_categories)
