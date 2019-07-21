import numpy as np
import matplotlib.pyplot as plt
from network.outliers_removal_network_single import ORNet
import cv2

# 数据
load_labeled = np.load("./data/test_data/16bit/test_data.npz")
test_descriptor = load_labeled["test_descriptor"]
test_label = load_labeled["test_label"]

# 声明网络
ornet = ORNet("./model/16bit/model")
predict = ornet.predict(test_descriptor)

# print(predict[124])
# print(test_label[124])
# print(test_descriptor[124])

predict = np.argmax(predict, axis=1)
test_label = np.argmax(test_label, axis=1)

outliers_removal_index = np.where(predict == 0)[0]
outliers_index = np.where(test_label == 1)[0]


# 统计去除误匹配数量
error = 0
error_removal = 0
for i in range(300):
    if test_label[i] == 1:
        error += 1
        if predict[i] == 1:
            error_removal += 1

# 统计正确匹配却认为是误匹配率：
right = 0
misremoval = 0
for i in range(300):
    if test_label[i] == 0:
        right += 1
        if predict[i] == 1:
            misremoval += 1


count = 0
diff_index = []
for i in range(300):
    if predict[i] == test_label[i]:
        count += 1
    else:
        diff_index.append(i)

print("正确率：", count / 300 * 100, "%")
print("误匹配剔除率：", error_removal /error * 100, "%")
print("误剔除率：", misremoval /right * 100, "%")

# 读点的数据
points_1 = np.load("./data/labeled_data/1_r.png_1_s.png_1.0.npz")["correspondence_label"][:2]
points_2 = np.load("./data/labeled_data/1_r.png_1_s.png_1.0.npz")["correspondence_label"][2:4]

# 读图的数据
img_1 = cv2.imread("./img/1_r.png")[:, :, [2, 1, 0]]
img_2 = cv2.imread("./img/1_s.png")[:, :, [2, 1, 0]]

img1 = cv2.resize(img_1, (800, 600))
img2 = cv2.resize(img_2, (800, 600))
h_img = np.hstack((img1, img2))


plt.figure(num="未去除误匹配")
plt.scatter(points_1[0, :], points_1[1, :], s=2, c="red")
plt.scatter(points_2[0, :] + 800, points_2[1, :],
            s=2, c="red")
plt.plot([points_1[0, :], points_2[0, :] + 800], [points_1[1, :], points_2[1, :]], linewidth=1, c="yellow")
plt.plot([points_1[0, outliers_index], points_2[0, outliers_index] + 800], [points_1[1, outliers_index], points_2[1, outliers_index]], linewidth=1, c="blue")
plt.imshow(h_img)

points_1 = points_1[:, outliers_removal_index]
points_2 = points_2[:, outliers_removal_index]
plt.figure(num="去除误匹配")
plt.scatter(points_1[0, :], points_1[1, :], s=2, c="red")
plt.scatter(points_2[0, :] + 800, points_2[1, :], s=2, c="red")
plt.plot([points_1[0, :], points_2[0, :] + 800], [points_1[1, :], points_2[1, :]], linewidth=1, c="yellow")
plt.imshow(h_img)

plt.show()