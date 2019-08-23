import numpy as np
import matlab.engine
import matlab
from utils import constant
from network.outliers_removal_network import ORNet
from descriptor_operation import create_descriptor, create_pre_matches
import time
#可以为所欲为的调用matlab内置函数
eng = matlab.engine.start_matlab()
eng.addpath('./matlab_transformation')
print("eng is ok")
# 启动
# 读图 产生sift描述子

img_path_r = "./img/16_r.png"
img_path_s = "./img/16_s.png"

#产生sift描述子 step_1
constant.SIFT_THRESHOLD = 0.4
pre_matches_r, pre_matches_s, des_1, des_2, partial_index_1, partial_index_2, img_1, img_2, resize_h, resize_w = \
    create_pre_matches.get_pre_matches(img_path_r, img_path_s, is_unique=True, k=None, k_num=None)

n_pre_matches_r, n_pre_matches_s, normal = eng.sirNorm2(matlab.double(pre_matches_s.tolist()),
                                                        matlab.double(pre_matches_r.tolist()), 2, nargout=3)

print(len(pre_matches_r))
print(len(partial_index_1))
# 声明网络
ornet = ORNet("./model/32bit/model")
# inlier初始化时为pre_match
inlier_index = np.arange(len(pre_matches_r))

for i in range(10):
    constant.ROTATION_SCALE_INV_SIFT_WEIGHTS = constant.ROTATION_SCALE_INV_SIFT_WEIGHTS
    # 产生我自己定义的描述子 step_2
    descriptor_categories = np.array([constant.ROTATION_SCALE_INV])
    print("KNN开始")
    # inlier也更新的版本
    descriptor_final = create_descriptor.create_descriptor_inlier_version(
        np.array(n_pre_matches_r), np.array(n_pre_matches_s), des_1, des_2, partial_index_1, partial_index_2,
        inlier_index, descriptor_categories=descriptor_categories)
    # inlier未更新的版本
    # descriptor_final = create_descriptor.create_descriptor(np.array(n_pre_matches_r), np.array(n_pre_matches_s), des_1, des_2, partial_index_1, partial_index_2,
    #     descriptor_categories=descriptor_categories)
    print("KNN结束")
    # network筛选出可靠的inlier
    predict = ornet.predict(descriptor_final)
    inlier_index = np.where(np.argmax(predict, axis=1) == 0)[0]
    print("inlier数量：", len(inlier_index))
    # 转换类型以便于喂入matlab +1 是因为matalb从1开始索引
    n_pre_matches_s, param = eng.featureTransform(n_pre_matches_s, n_pre_matches_r, matlab.int32((inlier_index+1).tolist()), nargout=2)

img_1 = img_1[:,:,[2,1,0]]
img_2 = img_2[:,:,[2,1,0]]
img_1 = np.array(img_1)/255.0
img_2 = np.array(img_2)/255.0
img_1 = matlab.double(img_1.tolist())
img_2 = matlab.double(img_2.tolist())
pre_matches_r = matlab.double(pre_matches_r.tolist())
pre_matches_s = matlab.double(pre_matches_s.tolist())
eng.demonstration(img_2, img_1 , pre_matches_s, pre_matches_r, matlab.int32((inlier_index+1).tolist()), nargout=0)
time.sleep(100)
