import unittest
import numpy as np
from auction import organic


class TestOrganic(unittest.TestCase):

    def test_shift_organic_for_alloc0(self):
        organic_idx = np.arange(10)
        alloc_pos = []
        r = organic.shift_organic_for_alloc(organic_idx, alloc_pos)
        self.assertListEqual(list(r), list(np.arange(10)))

    def test_shift_organic_for_alloc1(self):
        organic_idx = np.arange(10)
        alloc_pos = [0]
        r = organic.shift_organic_for_alloc(organic_idx, alloc_pos)
        self.assertListEqual(list(r.data), [-1] + list(np.arange(10)))

    def test_shift_organic_for_alloc2(self):
        organic_idx = np.arange(10)
        alloc_pos = np.arange(10)
        r = organic.shift_organic_for_alloc(organic_idx, alloc_pos)
        self.assertListEqual(list(r.data), [-1]*10 + list(organic_idx))

    def test_shift_organic_for_alloc3(self):
        organic_idx = np.arange(20)
        alloc_pos = [2, 6, 10, 14, 18]
        r = organic.shift_organic_for_alloc(organic_idx, alloc_pos)
        self.assertListEqual(
            list(r.data), [
                0, 1, -1, 2, 3, 4, -1, 5, 6, 7, -1, 8, 9, 10, -1,
                11, 12, 13, -1, 14, 15, 16, 17, 18, 19])

    def test_shift_priorities(self):
        alloc_idx = np.array([0, 1, 2])
        owner_ids = np.array([1, 1, 2, 1, 1, 2, 2, 3, 4, 5, 6])
        priorities = np.array([1, 0, 2, 2, 3, 0, 1, 0, 0, 0, 0])
        p2 = organic.shift_priorities(alloc_idx, owner_ids, priorities)
        self.assertListEqual([0, 1, 0, 2, 3, 1, 2, 0, 0, 0, 0], list(p2))

    def test_bid_in_organic(self):
        B = np.arange(100, dtype=np.float).reshape((10, 10)) + 1.
        organic_idx = np.ma.array(
            [3, -1, 4, 9, -1, 0, -1, -1, -1, 1], mask=[0,1,0,0,1,0,1,1,1,0], dtype=np.int)
        ins_to_imp_curve = np.array([1, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.05])
        B2 = organic.bid_in_organic(B, organic_idx, ins_to_imp_curve)
        # bid is zero for winner.
        self.assertListEqual(list(B2[organic_idx[0]]), [0.]*10)
        # bid is unchanged if item is not allocated
        self.assertListEqual(list(B[2]), list(B2[2]))
        # bid is scaled lower for organically-allocated item [1]
        i = organic_idx[2]
        v = ins_to_imp_curve - ins_to_imp_curve[2]
        v[v<0] = 0
        self.assertListEqual(list(v), [0.5, 0.25] + [0.]*8)
        self.assertListEqual(list(v * B[i]), list(B2[i]))
        # bid is scaled lower for organically-allocated item [1]
        i = organic_idx[5]
        v = ins_to_imp_curve - ins_to_imp_curve[5]
        v[v<0] = 0
        self.assertListEqual(list(np.round(v, 2)), [0.8, 0.55, 0.3, 0.2, 0.1, 0., 0., 0., 0., 0.])
        self.assertListEqual(list(v * B[i]), list(B2[i]))

    def test_quality_in_organic(self):
        Q = np.arange(100, dtype=np.float).reshape((10, 10)) - 100
        organic_idx = np.array([0, 3, 2, 1, 5])  # note, not all 10 positions allocated
        Q2 = organic.quality_in_organic(Q, organic_idx)
        # first allocated is all zero
        self.assertListEqual(list(Q2[organic_idx[0]]), [0.] * 10)
        # unallocated item is unchanged
        self.assertListEqual(list(Q2[6]), list(Q[6]))
        # item allocated lower on the page has less ENUE above that pos, 0 under it
        self.assertListEqual(list(Q2[organic_idx[-1]]), [-4.,-3.,-2.,-1.,] + [0.]*6)

    def test_get_organic_idx(self):
        a = np.array([-1, 0, -1, 1, -1, 3, -1, 2, -1, -1, -1])
        b = organic.get_organic_idx(a)
        self.assertListEqual(list(b), [1, 3, 7, 5])


if __name__ == '__main__':
    unittest.main()
