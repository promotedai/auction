import numpy as np
import unittest
from mockgen import unified as mockgen_uni
import auction
import page
from auction import assign
from auction import organic, assign


PAGE_N = 5


class TestAuction(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.A = mockgen_uni.get_items(page_size=PAGE_N, n=20)
        self.Page = page.make_env_curves(n=PAGE_N)
        self.PR = params.ParamCls()
        self.organic_idx = organic.get_organic_idx(self.A['organic_rank'].to_numpy())
        # positions reserved for promotions
        self.alloc = [0, 2, 3]

    def test_bid_tensor(self):
        # 500us for 1000x20
        BQ = auction.bid_tensor(self.A, self.Page, self.organic_idx, self.PR)
        self.assertEqual(BQ.shape, (2, 20, PAGE_N))
        # bid_tensor does not filter un-allocatable items
        self.assertFalse(np.all(np.any(BQ[0] >= self.PR.reserve_price, axis=1)))

    def test_pre_allocate(self):
        self.PR.reserve_price = 1. / 1000
        BQ, owner_priority, organic_idx, idx_map, owner_ids, reserves = auction.pre_allocate(self.A, self.Page, self.PR)
        # returned values are indexed after row-filtered (so, 10, versus 20)
        self.assertEqual(BQ.shape, (2, 10, PAGE_N))

        self.assertEqual(owner_priority.shape, (10,))
        self.assertEqual(organic_idx.shape, (PAGE_N,))
        self.assertEqual(owner_ids.shape, (10,))
        self.assertLess(np.max(organic_idx), 10)  # allocations are in filtered index space

        # remaining items must exceed reserve bid or be organically allocated
        can_be_allocated = np.any(BQ[0] >= self.PR.reserve_price, axis=1)
        v = np.array([False]*10)
        v[organic_idx] = True
        self.assertListEqual(
            list(can_be_allocated),
            [True, True, True, True, False, True, True, True, False, True])
        self.assertTrue(np.all(can_be_allocated | v))
        self.assertListEqual(list(self.organic_idx), list(idx_map[organic_idx]))
        # item 5 has an owner with 2 items
        self.assertListEqual(list(owner_priority), [0]*5 + [1] + [0]*4)
        self.assertEqual(len(set(owner_ids)), 9)
        d = dict(zip(*np.unique(owner_ids, return_counts=True)))
        self.assertEqual(d[owner_ids[5]], 2)
        # confirm that only and all zero reserve prices are organically allocated
        self.assertListEqual(list(reserves), [0.0, 0.001, 0.001, 0.001, 0.0, 0.001, 0.001, 0.0, 0.0, 0.0])
        self.assertListEqual(sorted(organic_idx), sorted(np.flatnonzero(reserves == 0)))

    def test_pre_allocate_raise_reserve(self):
        self.PR.reserve_price = 1. / 1000
        BQ, owner_priority, organic_idx, idx_map, owner_ids, reserves1 = auction.pre_allocate(self.A, self.Page, self.PR)
        max_bids = np.max(BQ[0], axis=1)
        self.assertEqual(np.sum(max_bids < 0.1), 5)
        self.assertEqual(np.max(reserves1), 0.001)

        # 7 and not 5 remaining because of organic allocations
        self.PR.reserve_price = 0.1
        BQ, owner_priority, organic_idx, idx_map, owner_ids, reserves2 = auction.pre_allocate(self.A, self.Page, self.PR)
        self.assertEqual(np.max(reserves2), 0.1)
        # returned values are indexed after row-filtered (so, 10, versus 20)
        self.assertEqual(BQ.shape, (2, 7, PAGE_N))
        self.assertEqual(owner_priority.shape, (7,))
        self.assertEqual(organic_idx.shape, (PAGE_N,))
        self.assertEqual(owner_ids.shape, (7,))
        self.assertLess(np.max(organic_idx), 7)
        self.assertListEqual(list(reserves2), [0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0])
        self.assertListEqual(sorted(organic_idx), sorted(np.flatnonzero(reserves2 == 0)))

    def test_pre_allocate_raise_exchange(self):
        # added test because exchangeRate was before passed incorrectly within pre_allocate
        self.PR.exchange_rate = 1.
        BQ1, owner_priority1, _, _, _, _ = auction.pre_allocate(self.A, self.Page, self.PR)
        self.PR.exchange_rate = 100000
        BQ2, owner_priority2,  _, _, _, _ = auction.pre_allocate(self.A, self.Page, self.PR)
        self.assertTrue(np.all(BQ1 == BQ2))
        # exchange rate only affects owner priority, not BQ. In this example, there aren't multiple owners

    def test_pre_allocate_top_k(self):
        self.PR.pre_allocate_top_k = 10
        BQ, owner_priority, organic_idx, idx_map, owner_ids, _ = auction.pre_allocate(self.A, self.Page, self.PR)
        max_bids = np.max(BQ[0], axis=1)
        self.assertEqual(np.sum(max_bids < 0.1), 2)

        self.assertEqual(BQ.shape, (2, 5, PAGE_N))
        self.assertEqual(owner_priority.shape, (5,))
        self.assertEqual(organic_idx.shape, (PAGE_N,))
        self.assertEqual(owner_ids.shape, (5,))
        self.assertLess(np.max(organic_idx), 5)

    def test_pre_allocate_top_k_disable(self):
        self.PR.pre_allocate_top_k = -1
        self.PR.reserve_price = 0
        BQ, owner_priority, organic_idx, idx_map, owner_ids, _ = auction.pre_allocate(self.A, self.Page, self.PR)
        self.assertEqual(BQ.shape, (2, 20, PAGE_N))

    def test_pre_allocate_top_k_organic(self):
        self.PR.pre_allocate_top_k = 0
        self.PR.reserve_price = 0
        BQ, owner_priority, organic_idx, idx_map, owner_ids, _ = auction.pre_allocate(self.A, self.Page, self.PR)
        self.assertEqual(BQ.shape, (2, PAGE_N, PAGE_N))
        self.assertListEqual(sorted(organic_idx), [0, 1, 2, 3, 4])

    def test_pre_allocate_top_k_lower(self):
        self.PR.pre_allocate_top_k = 6
        BQ, owner_priority, organic_idx, idx_map, owner_ids, _ = auction.pre_allocate(self.A, self.Page, self.PR)
        self.assertEqual(BQ.shape, (2, 5, PAGE_N))

    def test_pre_allocate_top_k_big(self):
        np.random.seed(1)
        A = mockgen_uni.get_items(page_size=PAGE_N, n=1000)
        self.PR.pre_allocate_top_k = 10
        BQ, owner_priority, organic_idx, idx_map, owner_ids, _ = auction.pre_allocate(A, self.Page, self.PR)
        # choose top 10 + 5 allocated.
        # this is not always necessarily true because of overlap with organic
        self.assertEqual(BQ.shape, (2, 14, PAGE_N))

    def test_pre_allocate_top_k_1(self):
        self.PR.pre_allocate_top_k = 1
        BQ, owner_priority, organic_idx, idx_map, owner_ids, _ = auction.pre_allocate(self.A, self.Page, self.PR)
        self.assertEqual(BQ.shape, (2, 5, PAGE_N))  # keep 5 organic allocated

    def test_alloc_BQ(self):
        BQ, owner_priority, organic_idx, idx_map, owner_ids, _ = auction.pre_allocate(
            self.A, self.Page, self.PR)
        alloc_idx = organic.shift_organic_for_alloc(organic_idx, self.alloc)
        alloc_idx_page = alloc_idx[:len(self.Page)]
        BQ2 = auction.alloc_BQ(
            BQ=BQ,
            alloc_idx_page=alloc_idx_page,
            owner_ids=owner_ids,
            owner_priority=owner_priority,
            ins_to_imp_curve=self.Page.impression.to_numpy(),
            Param=self.PR
        )
        self.assertEqual(BQ.shape, BQ2.shape)


    def test_run_to_solve(self):
        BQ, owner_priority, organic_idx, reverse_row_map, owner_ids, _ = auction.pre_allocate(
            self.A, self.Page, self.PR)
        alloc_idx = organic.shift_organic_for_alloc(organic_idx, self.alloc)
        alloc_idx_page = alloc_idx[:len(self.Page)]
        BQ_alloc = auction.alloc_BQ(
            BQ=BQ,
            alloc_idx_page=alloc_idx_page,
            owner_ids=owner_ids,
            owner_priority=owner_priority,
            ins_to_imp_curve=self.Page.impression.to_numpy(),
            Param=self.PR
        )
        U = BQ_alloc[0] + BQ_alloc[1] * self.PR.exchange_rate
        winners = assign.solve(U, self.alloc, utility_floor=self.PR.utility_floor)
        self.assertListEqual(list(winners), [0, 4, 6])
        w = reverse_row_map[winners]
        self.assertListEqual(list(w), [0, 12, 19])

    def test_run(self):
        auctionResponse = auction.run(self.A, self.Page, alloc=self.alloc, Param=self.PR)
        idx = auctionResponse.winner_idx.values
        prices = auctionResponse.price.values
        self.assertListEqual(self.alloc, [0, 2, 3])
        self.assertListEqual(list(idx), [0, 8, 12, 19, 13])
        # really? for this auction, the winners were already organically allocated in this order
        self.assertListEqual(list(idx), list(self.organic_idx))
        # because the winners were all already organically allocated, prices are all zero
        for i in prices:
            self.assertAlmostEqual(i, 0)

    def test_ranks(self):
        r = auction.ranks([0.2, 10, 3.3])
        self.assertListEqual(list(r), [0, 2, 1])

        r = auction.ranks([3.3, 10, 3.5, 100, -1])
        self.assertListEqual(list(r), [1, 3, 2, 4, 0])

    def test_ranks_zeros(self):
        r = auction.ranks([0, 10, 0, 0, 1.2])
        self.assertListEqual(list(r), [0, 4, 1, 2, 3])

    def test_toprank_U(self):
        U = np.array([
            [100, 0, 0],
            [90, 0, 0],
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [95, 4, 4],
        ])
        amax = np.argsort(-1 * np.max(U, axis=1))
        amin = np.argsort(-1 * np.min(U, axis=1))
        self.assertListEqual(list(amax), [0, 5, 1, 3, 4, 2])
        self.assertListEqual(list(amin), [5, 3, 4, 2, 0, 1])
        c = np.empty(amax.size * 2, dtype=int)
        # interleave
        c[0::2] = amax
        c[1::2] = amin
        self.assertListEqual(list(c), [0, 5, 5, 3, 1, 4, 3, 2, 4, 0, 2, 1])
        uniques, first_idx = np.unique(c, return_index=True)
        self.assertListEqual(list(uniques), [0, 1, 2, 3, 4, 5])
        self.assertListEqual(list(first_idx), [0, 4, 7, 3, 5, 1])
        v = auction.toprank_U(U)
        self.assertListEqual(list(v), [0, 3, 5, 2, 4, 1])

    def test_toprank_U_trival(self):
        U = np.array([
            [100, 100, 100],
            [90, 90, 90],
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [4, 4, 4],
        ])
        v = auction.toprank_U(U)
        self.assertListEqual(list(v), [0, 1, 5, 3, 4, 2])

    def test_sum_alloc_util(self):
        A = np.arange(100).reshape((20, 5))
        u = auction.sum_alloc_util(A, [0, 4, 9], [1, 0.5, 0.25], {0,4,9})
        self.assertEqual(u, 21*0.5 + 47*0.25)

    def test_sum_alloc_util_pivot(self):
        A = np.arange(100).reshape((20, 5))
        u = auction.sum_alloc_util(A, [0, 4, 9], [1, 0.5, 0.25], {})
        self.assertEqual(u, 0)

        A = np.arange(100).reshape((20, 5))
        u = auction.sum_alloc_util(A, [0, 4, 9], [1, 0.5, 0.25], {0})
        self.assertEqual(u, 0)

        A = np.arange(100).reshape((20, 5))
        u = auction.sum_alloc_util(A, [0, 4, 9], [1, 0.5, 0.25], {0, 4})
        self.assertEqual(u, 21*0.5)

    def test_move_imp_curve(self):
        curve = [1, 0.75, 0.5, 0.25, 0.1]
        curve_v = np.array(curve)
        self.assertListEqual(list(auction.move_imp_curve(curve_v, 0)), curve)
        self.assertListEqual(
            list(auction.move_imp_curve(curve_v, 1)),
            [1, 1, 0.5/0.75, 0.25/0.75, 0.1/0.75])
        self.assertListEqual(
            list(auction.move_imp_curve(curve_v, -1)),
            [1, 1, 1, 1, 1])

    def test_util(self):
        B = np.arange(100).reshape((20,5))
        Q = np.arange(100).reshape((20,5)) * -0.5
        r = np.zeros(20)
        U = auction.util(B, Q, 1, r)
        self.assertTrue(np.all(U == B*0.5))
        U = auction.util(B, Q, 2, r)
        self.assertTrue(np.all(U == 0))
        r[0] = 3
        r[-1] = 98
        U = auction.util(B, Q, 1, r)
        self.assertEqual(U[0, 0], 0)
        self.assertEqual(U[0, 4], 2)
        self.assertEqual(U[1, 0], 2.5)
        self.assertEqual(U[-1, 0], 0)
        self.assertEqual(U[-1, 4], 49.5)


if __name__ == '__main__':
    unittest.main()
