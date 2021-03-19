"""Given a utility matrix, solve for allocation."""
import numpy as np
from scipy.optimize import linear_sum_assignment
_EMPTY = -1000


def trim_right_empty(s):
    """Return a view excluding all EMPTY from the right. Assert no EMPTY."""
    i = len(s) - 1
    while i >= 0 and s[i] == _EMPTY:
        i -= 1
    ss = s[:i+1]
    if isinstance(ss, np.ndarray):
        assert np.all(ss != _EMPTY)
    else:
        assert _EMPTY not in ss
    return ss


def dequeue(s):
    """pop from front of list. efficent for numpy because s[1:] is a view, not a copy"""
    if len(s) == 0:
        return None, s
    return s[0], s[1:]


def fill_organic(organic, exclude, results, r_pos):
    """Helper to fill `results` with remaining `organic` items not in `exclude` """
    assert r_pos <= len(results)
    for idx in organic:
        if r_pos == len(results):
            return results
        if idx not in exclude:
            results[r_pos] = idx
            r_pos += 1
    return results


def solve_fill(U, alloc, ins_imp_curve, organic, page_size, utility_floor=0, exclude=None):
    """Wrapper for _solve_fill with default arguments."""
    assert isinstance(U, np.ndarray)
    if not isinstance(alloc, np.ndarray):
        alloc = np.array(alloc)
    if not isinstance(ins_imp_curve, np.ndarray):
        ins_imp_curve = np.array(ins_imp_curve)
    if not isinstance(organic, np.ndarray):
        organic = np.array(organic)
    assert page_size >= 0

    results = np.repeat(_EMPTY, page_size)
    if exclude is None:
        exclude = set()
    else:
        if not isinstance(exclude, set):
            exclude = set(exclude)

    _solve_fill(U, alloc, ins_imp_curve, exclude, organic, utility_floor, results)
    return trim_right_empty(results)


def _solve_fill(U, alloc, ins_imp_curve, no_promo, organic, utility_floor, results,
                r_pos=0, results_set=None):
    """Maximize assigned utility including organically-allocated items.
    """
    # Recursion does not branch, so we don't need to copy mutable data to preserve values.

    # Base case. No allocations. Fill with remaining organic items and return.
    if not alloc:
        return fill_organic(organic, [], results, r_pos)
    if results_set is None:
        results_set = set()

    # solve assignments. Because alloc is not empty, win_idx is not empty.
    win_idxs = solve(U, alloc, ins_imp_curve, utility_floor, exclude=no_promo)

    # fill results
    for i in range(r_pos, len(results)):

        # if this position is allocated for paid, try to fill with a winner
        paid_fill = None
        if len(alloc) and alloc[0] == i:
            winner, win_idxs = dequeue(win_idxs)
            if winner != _EMPTY:
                # Check if winner `e` had already been allocated in a higher position.
                # This can happen from organic back-filling.
                # This paid winner cannot be allocated again because it was already allocated.
                # Re-run allocation from this position to solve for a new paid winner,
                #   but exclude all already-allocated items from the paid auction.
                if winner in results_set:
                    return _solve_fill(
                        U=U, alloc=alloc, ins_imp_curve=ins_imp_curve,
                        no_promo=no_promo | results_set,  # no paid promo for anything already allocated.
                        organic=organic, utility_floor=utility_floor, results=results,
                        r_pos=i,  # start filling at the current position in results
                        results_set=results_set
                    )
                paid_fill = winner
            alloc = alloc[1:]

        # Fill the paid winner, or the next un-allocated organic item
        next_item = paid_fill
        if next_item is None:
            next_item, organic = dequeue(organic)
            # keep de-queuing items from organic until
            while next_item in results_set and len(organic) > 1:
                next_item, organic = dequeue(organic)

        # if still no next_item to allocate, then we're done.
        if next_item is None:
            return results

        # finally allocate this item and add it to the results_set
        results[i] = next_item
        results_set.add(paid_fill)


def solve(U, alloc=None, ins_imp_curve=None, utility_floor=None, exclude=None):
    """Wrapper for _solve with argument defaults and degenerate cases."""
    assert U.shape[1] > 0
    if alloc is None:
        alloc = np.arange(U.shape[1])
    else:
        assert len(alloc) > 0
    if ins_imp_curve is None:
        ins_imp_curve = np.repeat(1, U.shape[1])
    if utility_floor is None:
        utility_floor = 0

    if U.shape[0] == 0:
        # no items to assign, return _EMPTY for all columns
        return np.repeat(_EMPTY, len(alloc))

    return _solve(U, alloc, ins_imp_curve, utility_floor, exclude)


def _solve(U, alloc, ins_imp_curve, utility_floor, exclude=None):
    """Assign winners to each reserved position given a utility matrix.
    Args:
        U: float matrix of per position utility matrix of size (|items| x |page|)
        alloc: [int] of columns than can be allocated to
        ins_imp_curve: float vector of probability of impression given an insertion per pos, |page|
        utility_floor: minimum utility to be allocated (usually 0, maybe higher)
        exclude: [bool] vector of indices to never assign a position in allocation
    Returns:
        int vector of winner indices of U in order of `alloc` position index
            if unallocated, _EMPTY
    """
    # select only allocatable positions
    U = U[:, alloc]
    # remove rows that never exceed the utility floor
    keep = U.max(axis=1) > utility_floor
    # also remove all rows in exclude
    if exclude is not None:
        keep &= ~exclude
    reverse_map = np.where(keep)[0]
    U = U[keep]
    # apply ins_imp_curve
    U = U * ins_imp_curve[alloc]

    # Append zero items to prevent allocation of negative utility items
    # solutions with row indices over m are "no allocations"
    n = len(alloc)
    U = np.append(U, np.zeros((n, n)), axis=0)
    # newly appended rows point to index _EMPTY (not allocated)
    reverse_map = np.append(reverse_map, np.repeat(_EMPTY, n))

    # Run solver
    # 520us for 1000x20
    # 60us for 100x10
    win_row, win_col = linear_sum_assignment(U.max() - U)

    ordered_col = win_col.argsort()
    win_row = win_row[ordered_col]
    return reverse_map[win_row]


def fill(winners, alloc_idx, n):
    """Fill winners into page allocation.
    If missing winners, fill with organically allocated items.
    If an organically allocated item wins a higher slot, fill with organically allocated items.
    The same item should NEVER be allocated more than once!
    If there are not enough organic items, then fill with _EMPTY.
    Args:
        winners: [int] of idx from `solve.` |winners| = num -1 in alloc_idx. -1 is "unfilled."
        alloc_idx: [int] vector of idx of the organic allocation with reserved slots.
            -1 is reserved for promotion winners
            -2 is "empty". Don't fill with promotion winners and does not map to organics
        n: page size. n + |winners| = |alloc_idx|
    Returns:
        vector of int idx of final page allocations of len n. merging of winners and alloc_idx
    """
    assert len(alloc_idx) == len(winners) + n
    v = np.repeat(-2, n)
    win_i, alc_i, vi = 0, 0, 0
    filled = set()

    while vi < n:
        # WHAT IF alloc_idx has -1 or we go out of bounds?
        if alloc_idx[vi] != -1:
            while alloc_idx[alc_i] in filled or alloc_idx[alc_i] == -1:
                alc_i += 1
            v[vi] = alloc_idx[alc_i]
            alc_i += 1
        else:
            # try to fill with winner for this slot, else backfill with organics
            s = winners[win_i]
            win_i += 1
            if s != -1:
                if s not in filled:
                    v[vi] = s
                else:
                    while alloc_idx[alc_i] in filled or alloc_idx[alc_i] == -1:
                        alc_i += 1
                    v[vi] = alloc_idx[alc_i]
            else:
                # no winner fill from next alloc
                while alloc_idx[alc_i] in filled or alloc_idx[alc_i] == -1:
                    alc_i += 1
                v[vi] = alloc_idx[alc_i]
                alc_i += 1
        if v[vi] >= 0:
            filled.add(v[vi])
        vi += 1
    return v
