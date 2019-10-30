"""Implementation of the DBSCAN Clustering algorithm."""

import numpy as np


class DBSCAN:
    """Model for the classical query-based dbscan algorithm. 
       This version makes use of recursion to add points to the current cluster 
       until the stopping condition is met.
    """
    def __init__(self, eps=0.5, min_pts=3):
        """Initialize the model with the dbscan hyperparameters.
        
        Keyword Arguments:
            eps {float} -- The parameter specifying the radius of a neighborhood
                           wrt some point. (default: {0.5})
            min_pts {int} -- The minimum number of points within a points 
                             neightborhood required for it to be a core point.  (default: {3})
        """
        super(DBSCAN, self).__init__()
        self.eps = eps
        self.min_pts = min_pts  

    def fit(self, data):
        """The implementation of the dbscan algorithm for the input data.
        
        Arguments:
            data {List[List]} -- The input data for the algorithm in the form of a list of feature vectors.
        
        Returns:
            list -- Labels with the cluster number for each data point index.
        """
        self.data = data
        
        cluster_number = 0                     # default cluster number           
        self.labels = [0] * len(self.data)     # 0 means cluster is unassigned for the point 

        # Go through each point in the dataset
        for p in range(len(self.data)):
            if self.labels[p] > 0:             # if point is already assigned a cluster check next point
                continue 

            # else get the neighboring points of the point  
            neighbors = self._get_neighbors(self.data, p, self.eps)

            if len(neighbors) < self.min_pts:   # if point is noise check next point
                self.labels[p] = -1             # -1 means noise point
                continue
            
            # go to next cluster
            cluster_number += 1  

            # expand the cluster of current point with neighborhood points
            self._expand_cluster(p, neighbors, cluster_number)
                    
        return self.labels

    def _expand_cluster(self, p, neighbors, cluster_number):
        """Add neighborhood points to the current cluster until the stoppping
        condition is met.
        
        Arguments:
            p {List} -- The core point wrt which we are finding the cluster neighbors.
            neighbors {List[List]} -- The points in the neighborhood of the core point.
            cluster_number {int} -- The ID of the cluster that the core point belongs to.
        
        Returns:
            None -- No return values. Cluster labels are changed as side-effect.
        """
        # assign p to current cluster
        self.labels[p] = cluster_number  

        for q in neighbors: 
            if self.labels[q] < 0:                # if q was noise point change to border point
                self.labels[q] = cluster_number

            elif self.labels[q] == 0:             # if q is not part of a cluster assign it to cluster_number
                self.labels[q] = cluster_number
                neighbors_q = self._get_neighbors(self.data, q, self.eps)
                if len(neighbors_q) >= self.min_pts:
                    self._expand_cluster(q, neighbors.union(neighbors_q), cluster_number)
        return 0

    @staticmethod
    def _get_neighbors(data, center, eps):
        """Find the list of points in the neighborhood of a center point.
        
        Arguments:
            data {List[List]} -- The input data for the algorithm in the form of a list of feature vectors.
            center {List} -- The core point whose neighborhood is being calculated.
            eps {Float} -- The parameter specifying the radius of a neighborhood
                           wrt the center.
        
        Returns:
            List -- The list of the neighbors of the center point.
        """
        neighbors = set()
        for pt in range(0, len(data)):
            if np.linalg.norm(data[center] - data[pt]) < eps:    # distance metric is second norm
                neighbors.add(pt)
        return neighbors
