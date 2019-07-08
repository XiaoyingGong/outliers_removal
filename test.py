import numpy as np
import cv2

a = np.array([0.1, 0.8, 0.9, 0.7, 0.5])
if (a < 0.5).any() and (a >= 0.1).all():
    print("YES")
