import numpy as np
from .const import *


# 86 µs for 1000 items at 20 positions
def make_bids(A, n_pos, click_curve):
    """Matrix of all-position insertion bids for each position.
    Non-bidded items are expected to be removed prior this function.
    Args:
        A: dataframe of items. Requires columns:
            'bidType': enum of mockgen.const.BidType
               used for switching on how to compute bids
            'bid': float of bid for this optimization
            different bidTypes also depend on other columns.
                BidType_CPC: pClick
        n_pos: number of positions to bid on in the allocation
        click_curve: float vector used for BidType_CPC, % lower probability of click per position
            must be of length n_pos
    Returns:
        np float matrix of bid per item per position
        bidType BidType_NotPromoted has a bid of 0
    """
    assert n_pos == len(click_curve)
    assert isinstance(click_curve, np.ndarray)

    # CPM: flat bid for every position. Default bid.
    # 9us @ 10,000 items
    # NOTE: this includes all non-promoted items with bids of zero.
    B = np.broadcast_to(A['bid'].to_numpy(), shape=(n_pos, len(A))).transpose().copy()

    # CPC: bid per click, depends on position
    is_cpc = A['bidType'].to_numpy() == BidType_CPC
    CPC = np.outer(
        A['pClick'].to_numpy()[is_cpc] * A['bid'].to_numpy()[is_cpc], click_curve)
    B[is_cpc] = CPC
    return B


def is_promoted(A):
    """Return vector of promoted items from item list dataframe."""
    return A['bidType'].to_numpy() != BidType_NotPromoted
