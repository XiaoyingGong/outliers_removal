import numpy as np
import cv2

a = np.array([1, 3, 2, 5, 4, 3, 2, 3])
b = np.where((a > 1) & (a < 5))
b = np.array(b)
print(b)

b = b.reshape((len(b[0])))
print(b)