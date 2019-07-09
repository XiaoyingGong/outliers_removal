import numpy as np
import cv2

final_append = []
temp_append = []

a1 = np.array([1, 3, 2, 5, 4, 3, 2, 3])
a2 = np.array([1, 3, 2, 5, 4])

b1 = np.array([1, 3, 2, 6, 6, 6, 2, 6])
b2 = np.array([1, 2, 2, 3, 3])

temp_append.append(a1)
temp_append.append(a2)
final_append.append(temp_append)
temp_append = []
temp_append.append(b1)
temp_append.append(b2)
final_append.append(temp_append)
print(final_append)


