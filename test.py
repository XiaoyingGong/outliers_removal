import numpy as np
labeled_data_list = np.array(["1_r.png_1_s.png_1.0.npz", "2_r.png_2_s.png_1.0.npz", "3_r.png_3_s.png_1.0.npz",
                           "4_r.png_4_s.png_1.0.npz", "5_r.png_5_s.png_1.0.npz", "6_r.png_6_s.png_1.0.npz",
                           "7_r.png_7_s.png_1.0.npz", "8_r.png_8_s.png_1.0.npz", "9_r.jpg_9_s.jpg_1.0.npz",
                           "10_r.jpg_10_s.jpg_1.0.npz", "11_r.jpg_11_s.jpg_1.0.npz", "12_r.jpg_12_s.jpg_1.0.npz"])

load_labeled = np.load("./data/labeled_data/"+labeled_data_list[11])
# print(load_labeled["correspondence_label"][4][299])

load_labeled = np.load("./data/train_data/train_data.npz")
train_descriptor = load_labeled["train_descriptor"]
train_label = load_labeled["train_label"]
print(train_descriptor[299])
print(train_label[299])

