from mockgen import unified
from profilehooks import profile
import page, params, auction
import unittest

class ProfileAuction(unittest.TestCase):
    def test_profile_auction(self):
        N_ITEMS = 1000
        PAGE_SIZE = 20
        A = unified.get_items(n=N_ITEMS, page_size=PAGE_SIZE)
        Page = page.make_env_curves(PAGE_SIZE)
        Params = params.ParamCls()
        alloc = [0, 4, 8, 12, 16]
        for _ in range(500):
            wrapped_auction(A, Page, alloc, Params)

@profile
def wrapped_auction(*args, **kwargs):
    return auction.run(*args, **kwargs)

if __name__ == '__main__':
    unittest.main()
