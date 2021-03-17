import numpy as np
from scipy.special import expit


def softcap(v, offset, max_limit):
    """Apply a top-range sigmoid asymptotic maximum limit.
    Args:
        v: vector of floats to transform
        offset: below this value, values of v are unchanged
        max_limit: asymptote of highest value in transformed vector
    Returns:
         vector of floats corresponding to v with softcap applied
    """
    ramp = max_limit - offset
    return np.where(
        v < offset, v,
        offset + ((expit((v - offset) / (ramp / 2)) - 0.5) * ramp * 2))


# 400 µs for 1000 items x 20 positions
def make_neg_experience(quality_v, quality_thresh_curve, start, limit, is_promoted=None):
    """Matrix of expected negative user experience. Softcap quality to dynamic limit.
    Args:
        quality_v: float vector, per item raw quality score
        quality_thresh_curve: float vector, % position discount multiplier for quality
        start: float, apply softcapping to quality_v for values greater than `start` in first position
        limit: float, the asymptote limit for the 'maximum' quality score in first position
        is_promoted: bool vector if item is promoted. If not, bid, quality = 0. If None, ignore. |quality_v|
    Returns:
        np float matrix of expected negative experience per item per position.
            rows: items
            columns: positions
        Each element is negative and approaches 0. If near zero, then the item
        is of so high quality that we're confident that it is not a negative user
        experience.
    """
    # enforce type to avoid confusing pandas.Series errors
    assert type(quality_v) == type(quality_thresh_curve) == np.ndarray
    m = len(quality_v)
    n = len(quality_thresh_curve)

    # save computation on unpromoted items, which always have bid, ENUE, and utility of 0
    if is_promoted is not None:
        Q = np.zeros((m, n))
        Q2 = make_neg_experience(quality_v[is_promoted], quality_thresh_curve, start, limit)
        Q[is_promoted] = Q2
        return Q

    ramp = limit - start
    offset = (2 * start / (limit - start))
    ramp_v = ramp * quality_thresh_curve
    limit_v = limit * quality_thresh_curve
    start_v = start * quality_thresh_curve

    A = expit(np.outer(quality_v, 2 / ramp_v) - offset)
    # softcapped both at top and bottom.
    B = np.multiply((A - 0.5) * 2, ramp_v) + start_v
    # the item quality vector broadcast to all positions in quality_thresh_curve
    quality_M = np.broadcast_to(quality_v, (n, m)).transpose()
    C = np.where(B < start_v, quality_M, B)
    return C - limit_v
