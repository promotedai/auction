from auction import bid, comp_bid, organic, multi, assign
import numpy as np
import pandas as pd


def bid_tensor(A, Page, organic_idx, Param):
    """Return bid and ENUE (capped quality) for all items in A.
    NOTE: items in A that are neither organically allocated nor bidded can be excluded
        prior to calling this function for efficiency.
    Only computes allocation-pattern independent, "pre-allocation" bids and quality scores.
    B and Q will be manipulated depending on the promotion insertion pattern.

    Args:
        A: Dataframe of items with quality already computed
            + 'quality': raw quality score per item
            + 'organic_rank': int vector of organic rank
            + columns required for `bid.make_bids`
        Page: Dataframe for modeled page parameters
            + len() for number of positions
            + click for differential performance bidding by positions
            + quality_threshold for quality score to ENUE per position
        organic_idx: int vector of idx of A of organic ranks
        Param: object for system behavior parameters
    Returns:
        float tensor (2, |items|, |positions|) of bids and ENUE quality
   """
    # if organically allocated item is not promoted, then it's utility is always zero
    is_promoted = bid.is_promoted(A)

    B = bid.make_bids(A, len(Page), Page.click.to_numpy())
    Q = comp_bid.make_neg_experience(
        quality_v=A['quality'].to_numpy(),
        quality_thresh_curve=Page.quality_threshold.to_numpy(),
        start=Param.quality_start_limiting,
        limit=Param.quality_limit,
        is_promoted=is_promoted,
    )
    # If item would have been shown organically, lower its expected negative user experience
    #   do this before shifting items for promotion insertions (unlike for bids)
    Q = organic.quality_in_organic(
        Q,
        organic_idx,
        discount=Param.organic_alloc_qual_discount)
    return np.array([B, Q])


def ranks(v):
    """Convert v to rank order. e.g., [0.2, 10, 3.3] -> [0, 2, 1]
    Args:
        v: numeric vector
    Returns:
        [int] vector of ranks
    """
    a = np.argsort(v)
    x = np.arange(len(v))
    y = np.empty_like(x)
    y[a] = x
    return y


def toprank_U(U):
    """Rank in order of ability to allocate.
    We don't want to choose just the maximum, because we want to allocate
    the best option in lower positions where bids are lowest. We don't want to
    dominate top k by all high bids in the very top position.
    Args:
        U: float matrix
    Returns:
        [int] vector of row ranks, lower is top
    """
    amax = np.argsort(-1 * np.max(U, axis=1))
    amin = np.argsort(-1 * np.min(U, axis=1))
    c = np.empty(amax.size * 2, dtype=int)
    # interleave
    c[0::2] = amax
    c[1::2] = amin
    _, first_idx = np.unique(c, return_index=True)
    return ranks(first_idx)


def toprank_BQ(BQ, x):
    return toprank_U(BQ[0] + x * BQ[1])


def pre_allocate(A, Page, Param):
    """Sub-Workflow for all intermediate variables computed prior to an allocation pattern.
    Values returned exclude items that never pass the reserve bid
    Returns:
        BQ: float bid tensor with selected rows
        owner_priority: float vector of same owner priorities
        organic_idx: float vector of |Page| of filtered row indices per page position
        reverse_row_map: [int] position map back to original row index before row filtering
        owner_ids: vector of selected ownerIds from A
        reserve_prices: float vector of reserve prices from A: organically allocated items have reserve of 0
    """
    # organic_rank: size |A|. Allocated position of each item (or -1)
    # organic_idx: size |Page|. Row index of allocated item in each position.
    organic_rank = A['organic_rank'].to_numpy()
    organic_idx = organic.get_organic_idx(organic_rank)
    reserve_prices = np.repeat(Param.reserve_price, len(A))
    reserve_prices[organic_rank > -1] = 0

    BQ = bid_tensor(A, Page, organic_idx, Param)

    # items that never exceed the reserve bid and are not organically allocated will never
    # influence auction outcome and can be filtered for computational efficiency
    ever_exceeds_reserve = np.any((BQ[0].T >= reserve_prices).T, axis=1)
    could_be_organically_alloc = organic_rank != -1
    if Param.pre_allocate_top_k >= 0:
        top_k = toprank_BQ(BQ, Param.exchange_rate) < Param.pre_allocate_top_k
    else:
        top_k = np.repeat(True, len(could_be_organically_alloc))
    keep_rows = (ever_exceeds_reserve & top_k) | could_be_organically_alloc
    BQ = BQ[:, keep_rows]
    # we need new organic indices because we filtered rows
    organic_idx = organic.get_organic_idx(organic_rank[keep_rows])
    reverse_row_map = np.append(np.where(keep_rows)[0], -1)
    owner_ids = A[keep_rows]['ownerId'].to_numpy()
    reserve_price = reserve_prices[keep_rows]

    owner_priority = multi.owner_priority(
        B=BQ[0], Q=BQ[1], owner_ids=owner_ids, exchange_rate=Param.exchange_rate)

    # need an index of new row indices back to the original indices
    return BQ, owner_priority, organic_idx, reverse_row_map, owner_ids, reserve_price


# single allocation: 3.6ms, 20 positions, 7 alloc, 1000 items
# single allocation: 3.1ms, 5 positions, 2 alloc, 1000 items
# single allocation: 3.2ms, 5 positions, 2 alloc, 1000 items
# single allocation: 18.6ms, 20 positions, 7 alloc, 10,000 items
# single allocation: 2.8ms, 20 positions, 7 alloc, 100 items
# single allocation: 243ms, 20 positions, 7 alloc, 100,000 items
# single allocation: 831ms, 100 positions, 7 alloc, 100,000 items
# Note: after 10k items, will likely need some trimming prior to solving
def run(A, Page, alloc, Param):
    """Allocation workflow. Assign promoted items to an allocation pattern that maximizes value.
    Returns:
        [int] vector of A indices of length up to |Page|; the final allocation
        [float] vector of size |Page|, prices (if promoted) of winners of promoted auction
    """
    # pre-allocate filters some rows in A. Outputs are indexed by the new subset.
    # Use reverse_row_map to map back to these subset indices to the original index space
    BQ, owner_priority, organic_idx, reverse_row_map, owner_ids, reserve_prices = pre_allocate(A, Page, Param)
    n = len(Page)

    # ============ loop for specific allocation patterns ===============
    # DEPENDS ON ALLOCATION: change B and Q

    # alloc_idx is indices of (organically) allocated items prior to promoted allocation.
    alloc_idx = organic.shift_organic_for_alloc(organic_idx, alloc)

    # items allocated off the page aren't displayed.
    # If we have unfilled slots in auction, then we'll backfill organic items from the end
    # of alloc_idx back into ads slots.
    # This is why we keep alloc_idx available despite having alloc_idx_page.
    alloc_idx_page = alloc_idx[:n]
    imp_curve = Page.impression.to_numpy()

    # modify BQ by allocation-dependent changes
    BQ_alloc = alloc_BQ(BQ, alloc_idx_page, owner_ids, owner_priority, imp_curve, Param)
    B, Q = BQ_alloc[0], BQ_alloc[1]
    U = util(B, Q, Param.exchange_rate, reserve_prices)

    winners = assign.solve(U, alloc, imp_curve, Param.utility_floor, exclude=None)
    filled_winners = assign.fill(winners, alloc_idx.data, n)

    auction_result = pd.DataFrame({
        'winner_idx': reverse_row_map[filled_winners],
        'bid': np.diag(B[filled_winners]),
        'ownerPriority': owner_priority[filled_winners],
        'complementaryBid': np.diag(BQ[1][filled_winners, :]),
        'reserve_price': reserve_prices[filled_winners],
    })

    price_result = compute_prices(
        B, U, Param.utility_floor, n, imp_curve, alloc, alloc_idx, winners, filled_winners, owner_ids, reserve_prices)
    return pd.concat([auction_result, price_result], axis=1)


def alloc_BQ(BQ, alloc_idx_page, owner_ids, owner_priority, ins_to_imp_curve, Param):
    """Sub-workflow to adjust BQ bid matrices for an allocation pattern.
    Args:
        BQ: float bid tensor with selected rows
        alloc_idx_page:  [int] masked array of row idx in BQ allocated to each position. -1 is reserved
        owner_ids: vector of ownerIds corresponding to rows of BQ
        owner_priority: vector of int owner_priorities corresponding to rows of BQ
        ins_to_imp_curve: float vector of |Page| of probability of impression given an insertion per position
        Param: object for system behavior parameters (defaults to global singleton)
    Returns:
        copy of BQ with allocation-dependent modification
    """
    assert isinstance(ins_to_imp_curve, np.ndarray)
    # for the items that are actually allocated by the organic product...
    # ---
    # 1. Adjust owner priority.
    # This will penalize adding more items from promoters that are already organically allocated
    # If something is already organically allocated and is also promoted, then it will be first in
    # same-owner promotion priority, in order from most prominent position.
    alloc_owner_priority = organic.shift_priorities(
        alloc_idx_page, owner_ids, owner_priority)
    # 2. then, adjust quality by penalizing by priority
    Q = multi.same_owner_quality_penalize(
        BQ[1], alloc_owner_priority,
        additive=Param.same_promoter_additive,
        multi=Param.same_promoter_multi,
    )
    # 3. adjust bids for the items that are actually allocated by the organic product after insertions
    B = organic.bid_in_organic(BQ[0], alloc_idx_page, ins_to_imp_curve)
    return np.array([B, Q])


def util(B, Q, exchange_rate, reserves):
    """Compute utility.
    Args:
        B: float matrix of bids
        Q: float matrix of ENUE quality
        exchange_rate: float of conversion of Q to units of B ($)
        reserves: float vector of |B| of minimum price per item
    Returns:
        U: float matrix of utilities
    """
    U = B + Q * exchange_rate
    # cannot allocate if bid for a position is below the reserve
    bid_below_reserve = (B.T < reserves).T
    U[bid_below_reserve] = 0
    return U


def sum_alloc_util(U, filled_winners, pos_ins_imp, pivotal):
    """Return sum utility of allocation.
    Args:
        U: float matrix of utilities, (num items x page size)
        filled_winners: [int] of rows allocated to each position, size U.shape[1] == |page|
            idx must refer to rows in U
            -1 and -2 represent position placeholder non-allocations, they have utility of 0
        pos_ins_imp: [float] of probability of impression given an insertion, size U.shape[1] == |page|
        pivotal: only count idx in this set
    Returns:
        float of sum total utility for this allocation
    """
    s = 0
    for i, idx in enumerate(filled_winners):
        if idx < 0 or idx not in pivotal:
            continue
        s += U[idx, i] * pos_ins_imp[i]
    return s


def move_imp_curve(imp_curve, pos):
    """Set the current probability for a position of an impression to 100% and correct curve.
    Args:
        imp_curve: float vector of probabilities of impression given insertion from new page load
        pos: position last shown in the imp_curve
    Returns:
        [float] vector of corrected imp_curve for this position
    """
    v = imp_curve / imp_curve[pos]
    v[v > 1] = 1
    return v


# Timing: single allocation: 3.426ms, 20 positions, 6 alloc, 1000 items
def compute_prices(B, U, utility_floor, n, imp_curve, alloc, alloc_idx, winners, filled_winners, owner_ids, reserves):
    """
    Compute VCG price for each of the winners.
    Args:
        B: Bid Tensor
        U: float matrix of utilities, (num items x page size)
        utility_floor: Part of system behavior params denoted minimum utility for allocation
        n: page size. n + |winners| = |alloc_idx|
        imp_curve: float vector of probabilities of impression given insertion from new page load
        alloc: [int] vector of idx of reserved page positions for inserting promotions
        alloc_idx: [int] vector of idx of the organic allocation with reserved slots.
            -1 is reserved for promotion winners
            -2 is "empty". Don't fill with promotion winners and does not map to organics
        winners: [int] of idx from `solve.` |winners| = num -1 in alloc_idx. -1 is "unfilled."
        filled_winners: [int] of rows allocated to each position, size U.shape[1] == |page|
            idx must refer to rows in U
            -1 and -2 represent position placeholder non-allocations, they have utility of 0
        owner_ids: vector of selected ownerIds from A
        reserves: float vector of minimum (reserve) prices and minimum bids
    Returns:
        [Data Frame] with len(Page) rows and util: utility for winner at pos, altUtil: utility at pos without winner
            participating in auction
    """
    # COMPUTE PRICES
    result = pd.DataFrame(data={
        'util': np.zeros(n),
        'altUtil': np.zeros(n),
        'price': np.zeros(n),
        'reserved_price': np.zeros(n),
    })
    win_set = set(winners[winners >= 0])
    pos = 0
    for win_i, win in enumerate(winners):
        # 1. skip un-allocated winners
        if win == -1:
            continue

        # 2. find win position in filled_winners
        # winners can be filled at a higher position. Use the winner's filled position for utility pricing
        while filled_winners[pos] != win:
            pos += 1
            if pos >= len(filled_winners):
                raise Exception("Winner {} not in filled_winners {}".format(win, winners))
        # pos is now the actually allocated position of the current winner

        # 3. adjust ins_imp_curve by current position
        pos_ins_imp = move_imp_curve(imp_curve, pos)

        # 4. compute total expected value at this position
        sum_util = sum_alloc_util(U, filled_winners, pos_ins_imp, pivotal=win_set)

        # exclude from allocation all items owned by the current winner
        exclude = owner_ids[win] == owner_ids

        # re-allocate (using the original impression curve)
        alt_winners = assign.solve(U, alloc, imp_curve, utility_floor, exclude=exclude)
        filled_alt_winners = assign.fill(alt_winners, alloc_idx.data, n)
        alt_win_set = set(alt_winners[alt_winners >= 0])

        # compute new total expected value at this position
        # count the utility of the auction excluded items in case they're allocated organically
        alt_sum_util = sum_alloc_util(U, filled_alt_winners, pos_ins_imp, pivotal=alt_win_set)
        # alt_util should always <= util because we removed a winner.
        # it's not impossible for > to be true because util was allocated with a different
        # pos_ins_imp curve than was used to compute alt_util
        discount = max(sum_util - alt_sum_util, 0)
        # it's not impossible for b-q <= 0 because of fill shifting (but it's unlikely)
        # note that VCG price is: TotalUtilWithoutWinner - TotalUtilExceptForWinner
        price = B[win, pos] - discount
        reserve_price = reserves[win]

        result.at[pos, 'price'] = price
        result.at[pos, 'altUtil'] = alt_sum_util
        result.at[pos, 'util'] = sum_util
        result.at[pos, 'reserved_price'] = max(reserve_price, price)

    return result
