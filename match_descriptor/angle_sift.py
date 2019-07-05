import numpy as np


class AngleSift:
    '''
        input：
        pre_matches_1:预匹配后，sensed图片拥有的inliers
        per_matches_2:预匹配后，sensed图片拥有的inliers
        center_index_1:邻域的中心点在pre_matches_1中的下标
        center_index_2:邻域的中心点在pre_matches_2中的下标
        neighbor_index_1:领域的点的在pre_matches_1中的下标
        neighbor_index_2:领域的点的在pre_matches_2中的下标
        neighbor_dist_1:在点集2中领域的点距离中心点的距离
        neighbor_dist_2:在点集2中领域的点距离中心点的距离
    '''
    def __init__(self, pre_matches_1, per_matches_2, center_index_1, center_index_2,
                 neighbor_index_1, neighbor_index_2, neighbor_dist_1, neighbor_dist_2,
                 pre_matches_des_1, pre_matches_des_2):
        self.pre_matches_1 = pre_matches_1
        self.per_matches_2 = per_matches_2
        self.center_index_1 = center_index_1
        self.center_index_2 = center_index_2
        self.neighbor_index_1 = neighbor_index_1
        self.neighbor_index_2 = neighbor_index_2
        self.neighbor_dist_1 = neighbor_dist_1
        self.neighbor_dist_2 = neighbor_dist_2
        self.pre_matches_des_1 = pre_matches_des_1
        self.pre_matches_des_2 = pre_matches_des_2
        self.k = len(neighbor_index_1)

    '''
        主函数，用于创造sift_angle_descriptor，
        这个描述子的维度是根据领域的个数而定
        从2中开始找最近，逐个遍历
    '''
    def create_sift_angle_descriptor(self):
        sift_angle_descriptor = np.zeros(self.k)
        for i in range(self.k):
            if self.is_match_index(self.neighbor_index_2[i]) == 0: #是外点，就等于0
                sift_angle_descriptor[i] = 0.0
            else:#是内点，就计算夹角的差值
                sift_angle_descriptor[i] = 1.0
        return sift_angle_descriptor

    '''
        寻找是否在其中有与之匹配的index
        返回值为是否在另一个序列中有这个index(True or False)
    '''
    def is_match_index(self, index):
        return (self.neighbor_index_1 == index).any()

    '''
        计算角度
    '''
    def cal_angle(self):
        return
