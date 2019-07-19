import numpy as np
from utils import utils, constant

class IntraNeighborDist:
    def __init__(self, neighbor_dist_1, neighbor_dist_2):
        self.neighbor_dist_1 = neighbor_dist_1
        self.neighbor_dist_2 = neighbor_dist_2

    def cal_intra_neighbor_dist_descriptor(self):
        intra_neighbor_dist_descriptor = np.zeros(constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE)
        for i in range(constant.INTRA_NEIGHBOR_DIST_DESCRIPTOR_SIZE):
            intra_neighbor_dist_descriptor[i] = np.minimum(self.neighbor_dist_1[i], self.neighbor_dist_2[i]) \
                                                / np.maximum(self.neighbor_dist_1[i], self.neighbor_dist_2[i])
        return intra_neighbor_dist_descriptor


if __name__ == "__main__":
    a = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    b = np.array([1, 2, 1, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    intra_neighbor_dist = IntraNeighborDist(a, b)
    descriptor = intra_neighbor_dist.cal_intra_neighbor_dist_descriptor()
    print(descriptor)
