import numpy as np
from utils import utils, constant

class FuzzyGlobalCircle:
    '''
        在边上的，算两个都有，即都+1
        split：取同心圆的次数与个数
    '''
    def __init__(self, global_dist_1, global_dist_2, split):
        self.global_dist_1 = global_dist_1
        self.global_dist_2 = global_dist_2
        self.split = split
        self.len1 = len(self.global_dist_1)
        self.len2 = len(self.global_dist_2)

    def create_fuzzy_global_circle_descriptor(self):
        fuzzy_global_circle_descriptor = np.zeros(len(self.split))
        for i in range(len(self.split)):
            his_1 = self.count_points(self.global_dist_1, self.split[i])
            his_2 = self.count_points(self.global_dist_2, self.split[i])
            # print("his_1:", his_1, np.sum(his_1))
            # print("his_2:", his_2, np.sum(his_2))
            # 比较两个直方图
            fuzzy_global_circle_descriptor[i] = utils.chi_square(his_1, his_2)
            fuzzy_global_circle_descriptor[i] = utils.gaussian_penalty(fuzzy_global_circle_descriptor[i], constant.GAUSSIAN_PENALTY_SIGMA_2)
        return fuzzy_global_circle_descriptor

    # 计算每个环内的点的数量, k 代表分几个环
    def count_points(self, global_dist, k):
        len_his = np.zeros(k)
        split_len = global_dist[self.len1 - 1] / k
        count_len = 0
        for i in range(k):
            count_len += split_len
            if i != k - 1:
             len_his[i] = len([j for j in global_dist if count_len - split_len <= j <= count_len])
            else:
             len_his[i] = len([j for j in global_dist if count_len - split_len <= j <= global_dist[self.len1 - 1]])
        return len_his
