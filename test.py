import unittest

import numpy as np
from dbscan import DBSCAN


class TestDBSCAN(unittest.TestCase):
    def test_dbscan(self):
        """Test the dbscan algorithm on a small test data.
        """
        data = np.array([
            [1, 1.1],
            [1.2, 0.8],
            [0.8, 1],
            [3.7, 4],
            [3.9, 3.9],
            [3.6, 4.1],
            [10, 10]])
            
        clusters = DBSCAN(eps=0.5, min_pts=2).fit(data)
        self.assertEqual(clusters, [1, 1, 1, 2, 2, 2, -1])

if __name__ == "__main__":
    unittest.main()