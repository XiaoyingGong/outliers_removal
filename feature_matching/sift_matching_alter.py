import cv2
import numpy as np
import matplotlib.pyplot as plt

# 得到预匹配
# input: 两幅图像的路径
def get_matches(img_path1, img_path2, sift_threshold):
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
    kp1, des1 = get_key_point(img1)
    kp2, des2 = get_key_point(img2)
    good_match = get_good_match(des1, des2, sift_threshold)
    matching_points1, matching_points2 = get_matching_points(kp1, kp2, good_match)
    return matching_points1, matching_points2


# 得到匹配的点对，得到的为n*2的矩阵
def get_matching_points(kp1, kp2, good_match):
    matching_points1 = np.zeros((len(good_match), 2))
    matching_points2 = np.zeros((len(good_match), 2))
    n = len(good_match)
    for i in range(n):
        print(good_match[i])
        index1 = good_match[i][0].queryIdx
        index2 = good_match[i][0].trainIdx
        matching_points1[i, 0] = kp1[index1].pt[0]
        matching_points1[i, 1] = kp1[index1].pt[1]
        matching_points2[i, 0] = kp2[index2].pt[0]
        matching_points2[i, 1] = kp2[index2].pt[1]
    return matching_points1, matching_points2

# 得到关键点 input:图像 img
def get_key_point(img):
    #转换成灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


# 进行预匹配
def get_good_match(des1, des2, sift_threshold):
    # 特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    goodMatch = []
    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < sift_threshold * n.distance:
            goodMatch.append(m)
    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    return goodMatch

if __name__ == "__main__":
    get_matches("../img/1.png", "../img/2.png", 0.6)
