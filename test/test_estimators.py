import unittest

from fip.estimators import *


class TestPKLDivergenceEstimator(unittest.TestCase):
    def setUp(self):
        self.test_X = [('a', 'b', 'c', 'd'), ('a', 'b', 'x'), ('c', 'd'), ('a', 'b'), ('x', 'c'), ('a', 'x')]
        self.test_y = [1, 1, 1, 0, 0, 0]

    def test_fit(self):
        estimator = PKLDivergenceEstimator()
        estimator.fit(self.test_X, self.test_y)
        self.assertTrue(estimator.is_fitted())
        self.assertIsNotNone(estimator.pkld_profile)

    def test_predict(self):
        estimator = PKLDivergenceEstimator()
        estimator.fit(self.test_X, self.test_y)
        predictions = estimator.predict(self.test_X)
        self.assertEqual(len(predictions), len(self.test_y))
        self.assertSetEqual(set(predictions), {0, 1})

    def test_predict_proba(self):
        estimator = PKLDivergenceEstimator()
        estimator.fit(self.test_X, self.test_y)
        predictions = estimator.predict_proba(self.test_X)
        self.assertEqual(len(predictions), len(self.test_y))
        self.assertTrue(all(isinstance(p, float) for p in predictions))


if __name__ == '__main__':
    unittest.main()
