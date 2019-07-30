import numpy as np
from utils import utils, constant

# classname:RotationScaleInvDes（Rotation and Scale Invariance Descriptor）
# author：龚潇颖
# des：具有旋转与尺度不变性的描述子,分别为n维的角度，n维的长度之比
# input:  pre_matches_1:预匹配后，reference图片拥有的inliers
#         per_matches_2:预匹配后，sensed图片拥有的inliers
#         pre_matches_des_1:预匹配的描述子1
#         pre_matches_des_2:预匹配的描述子2
#         center_index_1:邻域的中心点在pre_matches_1中的下标
#         center_index_2:邻域的中心点在pre_matches_2中的下标
#         neighbor_index_1:领域的点的在pre_matches_1中的下标
#         neighbor_index_2:领域的点的在pre_matches_2中的下标
#         neighbor_dist_1:在点集2中领域的点距离中心点的距离
#         neighbor_dist_2:在点集2中领域的点距离中心点的距离
# output: 描述子
#
class RotationScaleInvDes:
    def __init__(self, pre_matches_1, pre_matches_2, pre_matches_des_1, pre_matches_des_2,
                 center_index_1, center_index_2,neighbor_index_1, neighbor_index_2,
                 neighbor_dist_1, neighbor_dist_2):
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
        shortest_index = -1
        degree_sift_des = np.zeros(self.k)
        scale_sift_des = np.zeros(self.k)
        for i in range(self.k):
            if not self.is_match_index(self.neighbor_index_2[i]): #是外点，就等于0
                degree_sift_des[i] = 0.0
                scale_sift_des[i] = 0.0
            else:#是内点，就计算夹角的差值，第一次计算到这儿，就是找最短边
                if not flag:
                    shortest_index = self.neighbor_index_2[i]
                    degree_sift_des[i] = 1.0
                    scale = self.neighbor_dist_2[i] / self.neighbor_dist_1[i]
                    scale_sift_des[i] = 1.0
                    flag = True
                    continue
                # 以下为非起始点的情况下计算degree_sift
                angle1, angle2 = self.get_degree(self.center_index_1, self.center_index_2, shortest_index,
                                                self.neighbor_index_2[i])
                des1, des2 = self.pre_matches_des_1[self.neighbor_index_2[i]], self.pre_matches_des_2[self.neighbor_index_2[i]]
                # 角度 + alpha * 描述子 alpha来控制sift描述子的强度
                degree_sift_1 = angle1 + des1 * constant.ROTATION_SCALE_INV_SIFT_WEIGHTS
                degree_sift_2 = angle2 + des2 * constant.ROTATION_SCALE_INV_SIFT_WEIGHTS
                sum_degree_sift_1 = np.sum(degree_sift_1)
                sum_degree_sift_2 = np.sum(degree_sift_2)
                degree_diff_result = np.minimum(sum_degree_sift_1, sum_degree_sift_2) / np.maximum(sum_degree_sift_1, sum_degree_sift_2)
                #将其放入结果向量中
                degree_sift_des[i] = degree_diff_result

                # 以下为非起始点的情况下计算len_sift
                cur_scale = self.neighbor_dist_2[i] / self.neighbor_dist_1[i]
                scale_diff_result = np.minimum(cur_scale, scale) / np.maximum(cur_scale, scale)
                scale_sift_des[i] = scale_diff_result
        return np.hstack((degree_sift_des, scale_sift_des))

    '''
        得到计算的两个角度
    '''
    def get_degree(self,  center_point_index_1, center_point_index_2, shortest_index, current_index):
        # 将中心点坐标 拿出来
        center_point_1 = self.pre_matches_1[center_point_index_1]
        center_point_2 = self.pre_matches_2[center_point_index_2]
        # 最短边的坐标拿出来，作为计算角度的参照
        ref_point_1 = self.pre_matches_1[shortest_index]
        ref_point_2 = self.pre_matches_2[shortest_index]
        # 最当前边的坐标拿出来,
        cur_point_1 = self.pre_matches_1[current_index]
        cur_point_2 = self.pre_matches_2[current_index]
        angle1 = self.cal_degree(center_point_1, ref_point_1, cur_point_1)
        angle2 = self.cal_degree(center_point_2, ref_point_2, cur_point_2)
        return angle1, angle2

    '''
    计算
    角度,都是锐角<180
    '''
    def cal_degree(self, center_point, ref_point, cur_point):
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
