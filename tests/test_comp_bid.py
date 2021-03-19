import numpy as np
import unittest
from auction import comp_bid


class TestCompBid(unittest.TestCase):

    def test_make_neg_experience_1d(self):
        qtc1 = np.array([1])
        quality_v = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        q1 = comp_bid.make_neg_experience(quality_v, qtc1, 1, 3)
        # rows: number of items in quality_v
        # columns: number of positions
        self.assertEqual(q1.shape, (len(quality_v), len(qtc1)))
        self.assertEqual([-3, -2, -1, 0, 1], list(q1[:5, 0] + 3))
        self.assertTrue(all(q1 < 0))

        q2 = comp_bid.make_neg_experience(quality_v, qtc1, 1, 6)
        self.assertEqual([-3, -2, -1, 0, 1], list(q2[:, 0][:5] + 6))
        self.assertGreater(q1[0, -1], q2[0, -1])

        q3 = comp_bid.make_neg_experience(quality_v, qtc1, -1, 3)
        self.assertEqual([-3, -2, -1], list(q3[:, 0][:3] + 3))
        self.assertLess(q3[3, 0] + 3, 0)
        self.assertLess(q3[4, 0] + 3, 1)

    def test_make_neg_experience(self):
        qtc1 = np.array([1, 0.5])
        quality_v = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        q1 = comp_bid.make_neg_experience(quality_v, qtc1, 1, 3)
        self.assertEqual(q1.shape, (len(quality_v), len(qtc1)))
        self.assertTrue(np.all(q1 < 0))

        qtc1_1 = np.array([1])
        qtc1_2 = np.array([0.5])
        q1_1 = comp_bid.make_neg_experience(quality_v, qtc1_1, 1, 3)
        q1_2 = comp_bid.make_neg_experience(quality_v, qtc1_2, 1, 3)
        self.assertTupleEqual(tuple(q1[:, 0]), tuple(q1_1))
        self.assertTupleEqual(tuple(q1[:, 1]), tuple(q1_2))

    def test_make_neg_experience_1d_promo(self):
        qtc1 = np.array([1])
        quality_v = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        is_promoted = quality_v % 2 == 0
        q1 = comp_bid.make_neg_experience(quality_v, qtc1, 1, 3, is_promoted)
        # rows: number of items in quality_v
        # columns: number of positions
        self.assertEqual(q1.shape, (len(quality_v), len(qtc1)))
        # items that are not promoted have ENUE of 0
        self.assertListEqual(list(q1[~is_promoted,0]), [0]*np.sum(~is_promoted))
        self.assertTrue(all(q1 <= 0))
        self.assertListEqual(
            list(q1[:, 0]),
            [0, -5, 0, -3, 0, -1.0757656854799804, 0, -0.18970349271026654, 0, -0.02677140369713893])

    def test_make_neg_experience_promo(self):
        qtc1 = np.array([1, 0.5])
        quality_v = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        is_promoted = quality_v % 2 == 0
        q1 = comp_bid.make_neg_experience(quality_v, qtc1, 1, 3, is_promoted)
        self.assertEqual(q1.shape, (len(quality_v), len(qtc1)))
        self.assertTrue(np.all(q1 <= 0))

        qtc1_1 = np.array([1])
        qtc1_2 = np.array([0.5])
        q1_1 = comp_bid.make_neg_experience(quality_v, qtc1_1, 1, 3, is_promoted)
        q1_2 = comp_bid.make_neg_experience(quality_v, qtc1_2, 1, 3, is_promoted)
        self.assertTupleEqual(tuple(q1[:, 0]), tuple(q1_1))
        self.assertTupleEqual(tuple(q1[:, 1]), tuple(q1_2))


if __name__ == '__main__':
    unittest.main()
