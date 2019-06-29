import cv2
import numpy as np
'''
 输入为两张图像的地址，返回值为指定阈值下的匹配
'''


def get_matches(img1_path, img2_path, sift_threshold):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    kp1, des1 = sift_kp(img1)
    kp2, des2 = sift_kp(img2)
    print(kp1[0].pt)
    tempMatrix = np.zeros((len(kp1), 2))
    # 去重
    for i in range(len(kp1)):
        tempMatrix[i][0] = kp1[i].pt[0]
        tempMatrix[i][1] = kp1[i].pt[1]
    print(len(tempMatrix))
    print(len(np.unique(tempMatrix, axis=0)))
    print(len(des2))
    print(len(np.unique(des2, axis=0)))
    good_match = get_good_match(des1, des2, sift_threshold)
    matching_points_1, matching_points_2 = get_matching_points(kp1, kp2, good_match)
    return matching_points_1, matching_points_2, kp1, kp2, good_match


# 得到在预匹配过后筛选的点,matching_points是一个n乘以2的二维矩阵，第一例为x坐标，第二例为y坐标
def get_matching_points(kp1, kp2, good_match):
    matching_points_1 = np.zeros((len(good_match), 2))
    matching_points_2 = np.zeros((len(good_match), 2))
    for i in range(len(good_match)):
        index1 = good_match[i].queryIdx
        index2 = good_match[i].trainIdx
        matching_points_1[i][0] = kp1[index1].pt[0]
        matching_points_1[i][1] = kp1[index1].pt[1]
        matching_points_2[i][0] = kp2[index2].pt[0]
        matching_points_2[i][1] = kp2[index2].pt[1]
    return matching_points_1, matching_points_2


# 得到关键点
def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    #kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp, des


# 做匹配
def get_good_match(des1, des2, sift_threshold):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < sift_threshold * n.distance:
            good.append(m)
    return good


if __name__ == "__main__":
    get_matches("../img/1.png", "../img/2.png", 0.6)
