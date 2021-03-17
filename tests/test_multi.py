import unittest
import numpy as np
from auction import multi


class TestMulti(unittest.TestCase):

    def test_get_rank_util(self):
        B = np.arange(0,9).reshape((3,3))
        Q = np.arange(0,9).reshape((3,3)) * 10
        rank_util = multi.get_rank_util(B, Q, exchange_rate=2)
        self.assertListEqual(list(rank_util), [21, 84, 147])
        self.assertIsInstance(rank_util, np.ndarray)

    def test_get_sameowner_priorities1(self):
        owner_ids = np.array(['aaa','bbb','bbb','ccc','aaa','aaa'])
        rank_utils = np.array([6,5,4,3,2,2])
        priorities = multi.get_sameowner_priorities(owner_ids, rank_utils)
        self.assertListEqual(list(priorities), [0.,0.,1.,0.,1.,2.])
        self.assertIsInstance(priorities, np.ndarray)
        rank_utils2 = np.array([1,2,3,4,5,5])
        priorities2 = multi.get_sameowner_priorities(owner_ids, rank_utils2)
        self.assertListEqual(list(priorities2), [2., 1., 0., 0., 0., 1.])
        self.assertIsInstance(priorities2, np.ndarray)

    def test_same_owner_quality_penalize(self):
        Q = np.arange(0,20).reshape((5,4))
        priorities = np.array([0,1,2,0,4])
        Q2 = multi.same_owner_quality_penalize(Q, priorities, -1, 1.5)
        expected = np.array([[0., 1., 2., 3.],
               [5., 6.5, 8., 9.5],
               [16., 18.25, 20.5, 22.75],
               [12., 13., 14., 15.],
               [77., 82.0625, 87.125, 92.1875]])
        self.assertListEqual(list(Q2.flatten()), list(expected.flatten()))

    def test_owner_priority(self):
        B = np.arange(0, 9).reshape((3, 3))
        Q = np.arange(0, 9).reshape((3, 3)) * 10
        owner_ids = np.array(['aaa', 'aaa', 'ccc'])
        priorities = multi.owner_priority(B, Q, owner_ids, 1)
        self.assertListEqual(list(priorities), [1, 0, 0])


if __name__ == '__main__':
    unittest.main()
