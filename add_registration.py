import matlab.engine
import matlab
import time
import numpy as np
from descriptor_operation import create_pre_matches
from utils import constant
import matplotlib.pyplot as plt
eng = matlab.engine.start_matlab()#可以为所欲为的调用matlab内置函数
eng.addpath('./matlab_src')
eng.addpath('./matlab_src/src')
print("eng is ok")


img_path_1 = "./img/14_r.jpg"
img_path_2 = "./img/14_s.jpg"
constant.SIFT_THRESHOLD = 0.6
pre_matches_1, pre_matches_2, des_1, des_2, partial_index_1, partial_index_2, img_1, img_2 = \
    create_pre_matches.get_pre_matches(img_path_1, img_path_2, is_unique=True, k=None, k_num=None)

pre_matches_1 = np.hstack((pre_matches_1, np.zeros([len(pre_matches_1), 1])))
pre_matches_2 = np.hstack((pre_matches_2, np.zeros([len(pre_matches_2), 1])))

plt.figure(num='Mpoints')
plt.scatter(pre_matches_1[:, 0], pre_matches_1[:, 1])
plt.figure(num='Fpoints')
plt.scatter(pre_matches_2[:, 0], pre_matches_2[:, 1])

Mpoints = matlab.double(pre_matches_1.tolist())
Fpoints = matlab.double(pre_matches_2.tolist())

result = eng.registration(Mpoints, Fpoints, nargout=1)
result = np.array(result)


plt.figure(num='result')
plt.scatter(result[:, 0], result[:, 1])
print(result)
plt.show()
