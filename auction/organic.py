"""Create slots and adjust priorities based on organic listings."""
import numpy as np


# 21.6 us for (np.arange(100), np.arange(0, 100, 5))
def shift_organic_for_alloc(organic_idx, alloc_pos):
    """Assign organic items after shifting to make space for promotion insertion allocations.
    organic_idx length must be enough to fill the page. Use -2 for "empty."
    Args:
        organic_idx: [int] of items assigned to positions without promotions
        alloc_pos: [int] of positions reserved for promotions in this allocation pattern
    Returns:
        tuple of:
            [int] of shifted organic item assignments with "-1" for reserved promo allocations
                of length |organic_idx| + |alloc_pos|
                -1 values are masked
    """
    # alloc_idxs need -1 per previously inserted item, because np.insert on a list is like
    # calling np.insert on each insertion in order in a loop. This accounts for each already
    # inserted item.
    k = len(alloc_pos)
    if k == 0:
        return organic_idx
    # this is not algorithmically efficient, but numpy is very fast and arrays are small, so this is OK
    v = np.insert(organic_idx, alloc_pos - np.arange(k), -1)
    return np.ma.masked_equal(v, -1, False)


def shift_priorities(alloc_idx, owner_ids, priorities):
    """Shift same-owner priorities to put organically-allocated items first.
    If multiple items from the same owner are allocated, then first positions have higher priority.
    NOTE: items that are not promoted still count in priority rankings.
    Args:
        alloc_idx: vector of row IDs allocated to which positions in allocation order
           idx of -1 indicates reserved slot for promotions
        owner_ids: vector of owner_ids, alloc_idx elements refer to positions in this vector
        priorities: vector of priorities, alloc_idx elements refer to positions in this vector
    Returns:
        priorities but adjusted to move allocated items to first priorities.
    """
    a = alloc_idx[alloc_idx >= 0]
    p2 = priorities.copy()
    # iterate on allocated owners and their items in reverse prominence order
    for own_id, idx in zip(owner_ids[a][::-1], a[::-1]):
        # select priorities of this owner and with a priority before than this item. increment
        # don't increment items with the same owner after this item because their order will not change.
        p2[(owner_ids == own_id) & (p2 < p2[idx])] += 1
        # put this item as the first priority
        p2[idx] = 0
    return p2


# 34us at (1000,20)
def bid_in_organic(B, organic_idx, ins_to_imp_curve):
    """Adjust bids given that some items may already be allocated organically.
    Args:
        B: matrix of bids, (item, position)
        organic_idx: [int] masked vector of item row idx (in B) to positions.
            Masked are promo reserved
        ins_to_imp_curve: [float] vector of prob of impression given insertion per pos
            note: columns in B == |ins_to_imp_curve|, which is the number of positions
    Returns:
        copy of B modified for organically allocated items
    """
    n = len(ins_to_imp_curve)
    assert B.shape[1] == n
    # C is how much more likely to be shown at col compared to row position
    # if col == row, then 0% more likely to be shown, because no change.
    C = np.broadcast_to(ins_to_imp_curve, (n, n))
    C = C - C.T
    C[C < 0] = 0
    B2 = B.copy()

    # replace bids for organically-allocated items with imp-ins weighted bids
    # in case the curve is longer than the allocated positions, trim the curve to the positions

    B2[organic_idx.compressed()] = \
        np.multiply(
            B2[organic_idx.compressed()],
            C[nomask_idx(organic_idx)])
    return B2


def nomask_idx(c):
    """Return idx of unmasked elements in `c`."""
    return (~c.mask).nonzero()


# 31us at (1000,20)
def quality_in_organic(Q, organic_idx, discount=1):
    """Adjust expected negative user experience given organic allocation.
    If an item was going to be organically allocated in a position, the negative
        impact of showing it in that position must be zero.
    Should be computed prior to shift-down for reserved promo allocation in organic_idx.
    Args:
        Q: ENUE matrix of (item, pos).quality after soft-max, not raw quality scores
        organic_idx: [int] vector of item row idx (in Q) to positions PRIOR to reserved alloc.
            NOT masked.
        discount: float parameter to further lower ENUE for organically allocated items
    Returns:
        copy of Q modified for organically allocated items
    """
    Q2 = Q.copy()
    # select ENUE of organically allocated items
    v = Q2[organic_idx]
    # the ENUE at the allocated pos is 0, and subtract that much from all other pos, too
    v -= np.broadcast_to(v.diagonal(), reversed(v.shape)).T
    v[v > 0] = 0  # ENUE is always non-positive
    # further lower the ENUE of already-allocated items when allocated to higher positions
    v *= discount
    Q2[organic_idx] = v
    return Q2


def get_organic_idx(organic_ranks):
    """Return vector of idx of A in organic ranks.
    Args:
        organic_ranks: int vector with ranks 0...n and unallocated items assigned -1
    Returns:
        int vector of row indices in A in order of organic allocation rank. exclude unallocated indices
    """
    idx = np.argsort(organic_ranks * -1)
    n = len(organic_ranks) - np.sum(organic_ranks == -1)
    return idx[::-1][-n:]
