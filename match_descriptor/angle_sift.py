import numpy as np
from utils import utils, constant

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
    def __init__(self, pre_matches_1, pre_matches_2, center_index_1, center_index_2,
                 neighbor_index_1, neighbor_index_2, neighbor_dist_1, neighbor_dist_2,
                 pre_matches_des_1, pre_matches_des_2):
        self.pre_matches_1 = pre_matches_1
        self.pre_matches_2 = pre_matches_2
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
        flag = False
        sum1 = 0
        sum2 = 0
        shortest_index = -1
        sift_angle_descriptor = np.zeros(self.k)
        # way2_sift_angle_descriptor = np.zeros(self.k)
        for i in range(self.k):
            if not self.is_match_index(self.neighbor_index_2[i]): #是外点，就等于0
                sift_angle_descriptor[i] = 0.0
            else:#是内点，就计算夹角的差值，第一次计算到这儿，就是找最短边
                if not flag:
                    shortest_index = self.neighbor_index_2[i]
                    sift_angle_descriptor[i] = 1.0
                    flag = True
                    continue
                angle1, angle2 = self.get_angle(self.center_index_1, self.center_index_2, shortest_index,
                                                self.neighbor_index_2[i])
                des1, des2 = self.pre_matches_des_1[self.neighbor_index_2[i]], self.pre_matches_des_2[self.neighbor_index_2[i]]
                # 计算angle_sift
                angle_sift_des1 = np.float32(des1 * angle1 / 180)
                angle_sift_des2 = np.float32(des2 * angle2 / 180)
                # 利用直方图的比较
                # angle_sift_des1 = np.resize(angle_sift_des1, [len(angle_sift_des1), 1])
                # angle_sift_des2 = np.resize(angle_sift_des2, [len(angle_sift_des2), 1])
                # hist_comp_result = utils.hist_correlation(angle_sift_des1, angle_sift_des2)
                # sift_angle_descriptor[i] = hist_comp_result
                # 计算angle_sift
                angle_sift_des1 = np.sum(des1) * angle1 / 180
                angle_sift_des2 = np.sum(des2) * angle2 / 180
                #Gaussian Penalty 归一到0到1
                x = np.minimum(angle_sift_des1, angle_sift_des2) / np.maximum(angle_sift_des1, angle_sift_des2)
                sift_angle_descriptor[i] = utils.gaussian_penalty(1 - x, constant.GAUSSIAN_PENALTY_SIGMA_1)
        return sift_angle_descriptor

    '''
        得到计算的两个角度
    '''
    def get_angle(self,  center_point_index_1, center_point_index_2, shortest_index, current_index):
        # 将中心点坐标 拿出来
        center_point_1 = self.pre_matches_1[center_point_index_1]
        center_point_2 = self.pre_matches_2[center_point_index_2]
        # 最短边的坐标拿出来，作为计算角度的参照
        ref_point_1 = self.pre_matches_1[shortest_index]
        ref_point_2 = self.pre_matches_2[shortest_index]
        # 最当前边的坐标拿出来,
        cur_point_1 = self.pre_matches_1[current_index]
        cur_point_2 = self.pre_matches_2[current_index]
        # print("center_point", center_point_1, center_point_2)
        # print("ref_point", ref_point_1, ref_point_2)
        # print("cur_point", cur_point_1, cur_point_2)
        angle1 = self.cal_angle(center_point_1, ref_point_1, cur_point_1)
        angle2 = self.cal_angle(center_point_2, ref_point_2, cur_point_2)
        # print("angle", angle1, angle2)
        # print()
        return angle1, angle2

    '''
    计算
    角度,都是锐角<180
    '''
    def cal_angle(self, center_point, ref_point, cur_point):
        dx1 = ref_point[0] - center_point[0]
        dy1 = ref_point[1] - center_point[1]
        dx2 = cur_point[0] - center_point[0]
        dy2 = cur_point[1] - center_point[1]
        angle1 = np.arctan2(dy1, dx1)
        angle1 = angle1 * 180 / np.pi
        angle2 = np.arctan2(dy2, dx2)
        angle2 = angle2 * 180 / np.pi
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    '''
        寻找是否在其中有与之匹配的index
        返回值为是否在另一个序列中有这个index(True or False)
    '''
    def is_match_index(self, index):
        return (self.neighbor_index_1 == index).any()
