import unittest

from fip.estimators import *


class TestPKLDivergenceEstimator(unittest.TestCase):
    def test_determine_classification_cutoff(self):
        test_X = [('a', 'b', 'c', 'd'), ('a', 'b', 'x'), ('c', 'd'), ('a', 'b'), ('x', 'c'), ('a', 'x')]
        test_y = [1, 1, 1, 0, 0, 0]
        estimator = PKLDivergenceEstimator()
        estimator.fit(test_X, test_y)


if __name__ == '__main__':
    unittest.main()
