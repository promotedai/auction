import unittest
import numpy as np
from auction import assign

E = assign._EMPTY

class TestAllocate(unittest.TestCase):

    def test_solve(self):
        # position 2 cannot be allocated
        # item 2 cannot be allocated
        A = np.array([[1, 1, -1, 3], [1, 3, -1, 1], [-1] * 4, [3, 1, -1, 1]])
        idxA = assign.solve(A)

        ans = [3, 1, E, 0]
        self.assertListEqual(list(idxA), ans)
        p = [1, 0, 3, 2]
        B = A[p]
        idxB = assign.solve(B)
        self.assertListEqual(list(idxB), [2, 0, E, 1])

    def test_solve2(self):
        # items 0 and 1 cannot be allocated
        A = np.array([
            [-1]*4, [-1]*4,
            [3, 2.99, 2.98, 2.97],
            [2, 1, 0.99, 0.98],
        ])
        idx = assign.solve(A)
        self.assertListEqual(list(idx), [3, 2, E, E])

    def test_solve_exclude(self):
        A = np.array([[1, 1, -1, 3], [1, 3, -1, 1], [-1] * 4, [3, 1, -1, 1]])
        idxA = assign.solve(A, exclude=np.arange(4) == 0)
        self.assertListEqual(list(idxA), [3, 1, E, E])
        idxAll = assign.solve(A, exclude=np.arange(4) > -1)
        self.assertListEqual(list(idxAll), [E]*4)

    def test_solve_zeros(self):
        self.assertListEqual(list(assign.solve(np.array([[0] * 4]))), [E] * 4)
        self.assertListEqual(list(assign.solve(np.array([[0, 1, 0, 0]]))), [E, 0, E, E])
        self.assertListEqual(list(assign.solve(np.array([[0] * 2]))), [E] * 2)

    def test_solve_empty(self):
        A = np.arange(100).reshape((20,5))
        self.assertListEqual(list(assign.solve(A[[False]*20])), [E] * 5)
        self.assertListEqual(list(assign.solve(np.zeros((0, 15)))), [E] * 15)
        with self.assertRaises(AssertionError):
            assign.solve(np.zeros((49, 0)))

    def test_solve_util_floor(self):
        A = np.array([
            [3, 2.99, 2.98, 2.97],
            [2, 1, 0.99, 0.98],
            [0.5, 0.5, 0.5, 0.5],
            [0.3, 0.3, 0.3, 0.3],
        ])
        idx = assign.solve(A, ins_imp_curve=None, utility_floor=0)
        self.assertListEqual(list(idx), [1, 0, 2, 3])
        idx = assign.solve(A, ins_imp_curve=None, utility_floor=0.6)
        self.assertListEqual(list(idx), [1, 0, E, E])
        # ins_imp_curve no longer affects utility floor trimming
        idx = assign.solve(A, ins_imp_curve=np.array([1, 0.6, 0.6, 0.6]), utility_floor=0.6)
        self.assertListEqual(list(idx), [1, 0, E, E])
        idx = assign.solve(A, ins_imp_curve=np.array([1, 0.6, 0.6, 0.1]), utility_floor=0.6)
        self.assertListEqual(list(idx), [1, 0, E, E])

    def test_solve_imp_curve(self):
        # not having an imp curve may not assign top weights first
        A = np.array([
            [1] * 4,
            [2] * 4,
            [4] * 4,
            [3] * 4,
        ])
        idx = assign.solve(A)
        best = [2, 3, 1, 0]
        # this seems to be an implementation quirk? assignments could be in any order
        self.assertListEqual(list(idx), best)
        # force allocation in this order
        idx2 = assign.solve(A, ins_imp_curve=np.array([0.5, 0.75, 2, 0.1]))
        self.assertListEqual(list(idx2), [1, 3, 2, 0])
        # test allocation and ins_imp_curve interaction
        idx3 = assign.solve(A, alloc=np.array([1, 2, 3]), ins_imp_curve=np.array([0.5, 0.75, 2, 0.1]))
        self.assertListEqual(list(idx3), [3, 2, 1])

    def test_fill_backfill(self):
        winners = [0, 1]
        alloc_idx = [-1, 2, -1, 10, 20]
        v = assign.fill(winners, alloc_idx, 3)
        # insert as normal
        self.assertListEqual(list(v), [0, 2, 1])

    def test_fill_backfill2(self):
        winners = [0, 10]
        # original allocation: 2, 10, 20
        alloc_idx = [-1, 2, -1, 10, 20]
        v = assign.fill(winners, alloc_idx, 3)
        # insert as normal. 10 won a higher slot, but it may also be back-filled
        self.assertListEqual(list(v), [0, 2, 10])

    def test_fill_backfill_organic_up(self):
        winners = [0, 1]
        alloc_idx = [-1, 0, -1, 10, 20]
        v = assign.fill(winners, alloc_idx, 3)
        # 0 won, so next organic, 10, was back-filled in. Then, 1.
        self.assertListEqual(list(v), [0, 10, 1])

    def test_fill_backfill_organic_up2(self):
        # original allocation was [0, 10, 20]
        winners = [0, 10]
        alloc_idx = [-1, 0, -1, 10, 20]
        v = assign.fill(winners, alloc_idx, 3)
        # do we want this outcome? what about filling slot 3 with a promotion?
        self.assertListEqual(list(v), [0, 10, 20])

    def test_fill_backfill_organic_up3(self):
        # original allocation was [0, 10, 20]
        winners = [1, 10]
        alloc_idx = [-1, 0, -1, 10, 20]
        v = assign.fill(winners, alloc_idx, 3)
        self.assertListEqual(list(v), [1, 0, 10])

    def test_fill_backfill_organic_missing_winner(self):
        # original allocation was [0, 10, 20]
        winners = [1, -1]
        alloc_idx = [-1, 0, -1, 10, 20]
        v = assign.fill(winners, alloc_idx, 3)
        self.assertListEqual(list(v), [1, 0, 10])

    def test_fill_backfill_missing_winner2(self):
        winners = [10, -1]
        alloc_idx = [-1, -1, 10, 20]
        v = assign.fill(winners, alloc_idx, 2)
        self.assertListEqual(list(v), [10, 20])

    def test_fill_empties(self):
        winners = [10, -1]
        alloc_idx = [-1, -1, -2, -2]
        v = assign.fill(winners, alloc_idx, 2)
        self.assertListEqual(list(v), [10, -2])
        v2 = assign.fill([-1, -1], alloc_idx, 2)
        self.assertListEqual(list(v2), [-2, -2])
        v3= assign.fill([-1], [-1, -2, -2], 2)
        self.assertListEqual(list(v3), [-2, -2])
        v3 = assign.fill([-1], [-1, -2], 1)
        self.assertListEqual(list(v3), [-2])


class TestUtil(unittest.TestCase):
    def test_deque_numpy(self):
        # confirm that deque returns a view of a numpy array, not a copy
        s = np.arange(5)
        i, ss = assign.dequeue(s)
        self.assertEqual(0, i)
        self.assertListEqual(list(s), [0, 1, 2, 3, 4])
        self.assertListEqual(list(ss), [1, 2, 3, 4])
        ss[-1] = 9
        s[-2] = 8
        self.assertListEqual(list(s), [0, 1, 2, 8, 9])
        self.assertListEqual(list(ss), [1, 2, 8, 9])

    def test_deque_native(self):
        # deque still returns a copy for native python lists
        s = [0, 1, 2, 3, 4]
        i, ss = assign.dequeue(s)
        self.assertEqual(0, i)
        self.assertListEqual(list(s), [0, 1, 2, 3, 4])
        self.assertListEqual(list(ss), [1, 2, 3, 4])
        ss[-1] = 9
        s[-2] = 8
        self.assertListEqual(list(s), [0, 1, 2, 8, 4])
        self.assertListEqual(list(ss), [1, 2, 3, 9])

    def test_fill_organic(self):
        s = assign.fill_organic([1, 2, 3], [], [0, 0, 0], 0)
        self.assertListEqual(s, [1, 2, 3])
        s = assign.fill_organic([1, 2, 3], [], [0, 0, 0], 2)
        self.assertListEqual(s, [0, 0, 1])
        s = assign.fill_organic([1, 2, 3], [1, 2], [0, 0, 0], 2)
        self.assertListEqual(s, [0, 0, 3])

    def test_trim_right_empty(self):
        E = assign._EMPTY
        s = assign.trim_right_empty([0, 1, 2, 3])
        self.assertListEqual(s, [0, 1, 2, 3])
        s = assign.trim_right_empty([0, 1, 2, E])
        self.assertListEqual(s, [0, 1, 2])
        s = assign.trim_right_empty([0, 1, E, E])
        self.assertListEqual(s, [0, 1])
        s = assign.trim_right_empty(np.array([0, 1, 2, 3]))
        self.assertListEqual(list(s), [0, 1, 2, 3])
        s = assign.trim_right_empty(np.array([0, 1, E, E]))
        self.assertListEqual(list(s), [0, 1])
        s = assign.trim_right_empty(np.array([E, E, E, E]))
        self.assertListEqual(list(s), [])
        # do not allow empties in other positions from the right
        with self.assertRaises(AssertionError):
            assign.trim_right_empty([E, 1, E, E])
        with self.assertRaises(AssertionError):
            assign.trim_right_empty(np.array([E, 1, E, E]))


if __name__ == '__main__':
    unittest.main()
