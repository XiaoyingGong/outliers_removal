import numpy as np
from sklearn.neighbors import NearestNeighbors


class K_NearestNeighbors:
    def __init__(self, train_data):
        self.train_data = train_data
        self.nn = NearestNeighbors(algorithm='kd_tree').fit(self.train_data)

    def get_k_neighbors(self, aim_point, k):
        # 没找到足量的点 就一直循环
        k_new = k + 1
        # while True:
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k_new)
        #去除掉自己
        zero_index = np.array(np.where(k_nearest_neighbors_dist[0] == 0.))
        k_nearest_neighbors_dist = np.delete(k_nearest_neighbors_dist, zero_index)
        k_nearest_neighbors_index = np.delete(k_nearest_neighbors_index, zero_index)
            #print(len(k_nearest_neighbors_dist[0]), " ", len(k_nearest_neighbors_index[0]))
            # if len(k_nearest_neighbors_dist) != k or len(k_nearest_neighbors_index) != k:
            #     k_new = k_new + 1
            # else:
            #     break
        return k_nearest_neighbors_dist, k_nearest_neighbors_index

