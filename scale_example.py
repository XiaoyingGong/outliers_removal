import matplotlib.pyplot as plt
import numpy as np
import cv2
from feature_matching import sift_matching

img_path1 = "./img/scale_test/bigger.png"
img_path2 = "./img/scale_test/smaller.png"
img1 = cv2.imread(img_path1)[:, :, [2, 1, 0]]
img2 = cv2.imread(img_path2)[:, :, [2, 1, 0]]
img2 = cv2.resize(img2, (800, 600))
sift_threshold = 0.3

pre_matches1, pre_matches2, des1, des2, match_index = sift_matching.get_matches(img1, img2, sift_threshold)
print(len(pre_matches1))
print(len(pre_matches2))
pre_matches1_t = np.transpose(pre_matches1)
pre_matches2_t = np.transpose(pre_matches2)

h_img = np.hstack((img1, img2))
plt.figure(num='scale_test')
plt.axis("off")
plt.imshow(h_img)
# 往上画
plt.scatter(pre_matches1_t[0], pre_matches1_t[1], s=2, c='red')
plt.scatter(pre_matches2_t[0] + 800, pre_matches2_t[1], s=2, c='red')

plt.show()


