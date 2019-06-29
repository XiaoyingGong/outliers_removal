import numpy as np

a = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5])
b = np.delete(a, [0, 1])
print(b)