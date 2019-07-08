import numpy as np
import cv2
# img1 = cv2.imread("./img/1.png")
# his1 = np.array([[1], [2], [3], [4], [5], [6]], dtype=np.float32)
# his2 = np.array([[1], [2], [3], [5], [5], [6]], dtype=np.float32)
# print(his1)
# rgbHist_1=np.zeros([16*16*16,1],np.float32)
# rgbHist_2=np.zeros([16*16*16,1],np.float32)
# result = cv2.compareHist(his1, his2, cv2.HISTCMP_CORREL)
# print(result)
his1 = np.array([1, 2, 3, 4, 5])
print(his1)
his1 =np.resize(his1, (5, 1))
print(his1)