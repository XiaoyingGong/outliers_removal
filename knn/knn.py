import numpy as np
from sklearn.neighbors import NearestNeighbors


class K_NearestNeighbors:
    def __init__(self, train_data):
        self.train_data = train_data
        self.nn = NearestNeighbors(algorithm='kd_tree').fit(self.train_data)

    def get_k_neighbors(self, aim_point, k, index):
        #print("aim_point:", aim_point, " k:", k, " index:", index)
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k+1)
        #去除掉自己
        zero_index = np.array(np.where(k_nearest_neighbors_dist == 0.))[0]
        k_nearest_neighbors_dist = k_nearest_neighbors_dist[k_nearest_neighbors_dist != 0.]
        k_nearest_neighbors_index = np.delete(k_nearest_neighbors_index, zero_index)
        print(len(k_nearest_neighbors_dist), " ", len(k_nearest_neighbors_index))
        return k_nearest_neighbors_dist, k_nearest_neighbors_index

