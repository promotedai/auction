import numpy as np
import pandas as pd
import unittest
import auction.bid as bid


class TestBid(unittest.TestCase):

    def setUp(self):
        self.A = pd.DataFrame({
            'bidType': [0, 1, 2, 0, 1, 2, 2],
            'bid': [1., 2., 3., 4., 5., 6., 7.],
            'pClick': [0, 0, 0.1, 0, 0, 0.2, 0.3],
        })
        self.click_curve = np.array([1, 0.75, 0.5, 0.5])

    def test_make_bids(self):
        B = bid.make_bids(self.A, len(self.click_curve), self.click_curve)
        self.assertTupleEqual(B.shape, (7, 4))
        self.assertTupleEqual(tuple(B[0]), (1.,) * 4)
        self.assertTupleEqual(tuple(B[1]), (2.,) * 4)
        self.assertAlmostEqual(B[2, 0], 0.3)
        self.assertAlmostEqual(B[2, 1], 0.3 * 0.75)
        self.assertAlmostEqual(B[2, 2], 0.3 * 0.5)
        self.assertAlmostEqual(B[2, 3], 0.3 * 0.5)
        self.assertTupleEqual(tuple(B[4]), (5.,) * 4)

    def test_is_promoted(self):
        v = bid.is_promoted(self.A)
        self.assertListEqual(list(v), [False, True, True, False, True, True, True])


if __name__ == '__main__':
    unittest.main()
