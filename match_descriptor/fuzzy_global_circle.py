import numpy as np
from utils import utils, constant
import matplotlib.pyplot as plt


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
        # 用于绘制
        self.neighbor_point_index = []

    def create_fuzzy_global_circle_descriptor_v0(self):
        fuzzy_global_circle_descriptor = np.zeros(len(self.split))
        for i in range(len(self.split)):
            his_1 = self.count_points(self.global_dist_1, self.split[i])
            his_2 = self.count_points(self.global_dist_2, self.split[i])
            # 比较两个直方图
            # chi-square 比较
            # fuzzy_global_circle_descriptor[i] = utils.chi_square(his_1, his_2)
            # fuzzy_global_circle_descriptor[i] = utils.gaussian_penalty(fuzzy_global_circle_descriptor[i], constant.GAUSSIAN_PENALTY_SIGMA_2)
            # 相关性比较
            his_1 = np.resize(his_1, (len(his_1), 1))
            his_2 = np.resize(his_2, (len(his_2), 1))
            plt.show()
            hist_comp_result = utils.hist_correlation(his_1, his_2)
            fuzzy_global_circle_descriptor[i] = utils.gaussian_penalty(hist_comp_result,
                                                                       constant.GAUSSIAN_PENALTY_SIGMA_2)
        return fuzzy_global_circle_descriptor, self.neighbor_point_index

    def create_fuzzy_global_circle_descriptor(self):
        circle_descriptor = np.zeros(len(self.split))
        fuzzy_circle_descriptor = np.zeros(len(self.split))
        for j in range(len(self.split)):
            his_1 = self.count_points(self.global_dist_1, self.split[j])
            his_2 = self.count_points(self.global_dist_2, self.split[j])

            # 计算模糊高斯计数策略的圆环
            fuzzy_his_1 = np.zeros(len(his_1), dtype=np.float32)
            fuzzy_his_2 = np.zeros(len(his_1), dtype=np.float32)

            for i in range(len(his_1)):
                if i == 0:
                    fuzzy_his_1[i] = his_1[i] + his_1[i + 1] * utils.gaussian_weight(i + 1, 1)
                    fuzzy_his_2[i] = his_2[i] + his_2[i + 1] * utils.gaussian_weight(i + 1, 1)
                elif i == (len(his_1) - 1):
                    fuzzy_his_1[i] = his_1[i - 1] * utils.gaussian_weight(i + 1, 1) + his_1[i]
                    fuzzy_his_2[i] = his_2[i - 1] * utils.gaussian_weight(i + 1, 1) + his_2[i]
                else:
                    fuzzy_his_1[i] = his_1[i - 1] * utils.gaussian_weight(i - 1, 1) + his_1[i] + his_1[
                        i + 1] * utils.gaussian_weight(i + 1, 1)
                    fuzzy_his_2[i] = his_2[i - 1] * utils.gaussian_weight(i - 1, 1) + his_2[i] + his_2[
                        i + 1] * utils.gaussian_weight(i + 1, 1)

            his_1 = np.resize(his_1, (len(his_1), 1))
            his_2 = np.resize(his_2, (len(his_1), 1))

            fuzzy_his_1 = np.resize(fuzzy_his_1, (len(fuzzy_his_1), 1))
            fuzzy_his_2 = np.resize(fuzzy_his_2, (len(fuzzy_his_2), 1))

            hist_comp_result = utils.hist_correlation(his_1, his_2)
            fuzzy_hist_comp_result = utils.hist_correlation(fuzzy_his_1, fuzzy_his_2)

            circle_descriptor[j] = hist_comp_result
            fuzzy_circle_descriptor[j] = fuzzy_hist_comp_result
        return circle_descriptor, fuzzy_circle_descriptor

    # 计算每个环内的点的数量, k 代表分几个环
    def count_points(self, global_dist, k):
        points_his = np.zeros(k, dtype=np.float32)
        split_len = global_dist[self.len1 - 1] / k
        count_len = 0
        index = None
        temp_append = []
        for i in range(k):
            count_len += split_len
            if i != k - 1:
                #points_his[i] = len([j for j in global_dist if count_len - split_len <= j <= count_len])
                index = np.array(np.where((global_dist >= count_len - split_len) & (global_dist <= count_len)))
                index = index.reshape([len(index[0])])
                points_his[i] = len(index)
            else:
                # points_his[i] = len([j for j in global_dist if count_len - split_len <= j <= global_dist[self.len1 - 1]])
                index = np.array(np.where((global_dist >= count_len - split_len) & (global_dist <= global_dist[self.len1 - 1])))
                index = index.reshape([len(index[0])])
                points_his[i] = len(index)
            temp_append.append(index)
        self.neighbor_point_index.append(temp_append)
        return points_his
