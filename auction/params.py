"""Global parameter manager, used for exploration.

Note on default parameter values in Python:
=====
Default argument values are set at initialization, not at function call time.
If using parameter exploration functionality, always explicitly pass the function arguments
    using a new call to the PR singleton versus using the argument default values.
"""


class ParamCls:
    def __init__(self):
        # --------------------------
        # REQUIRED AUCTION PARAMETERS
        # --------------------------
        # These parameters are always used. Set default values for testing and
        # code intellisense purposes
        # ==========================
        # Used in soft-cap of raw quality. Below this value, there is no soft-capping.
        # dft value about 10% percentile for mockgen default quality score
        self.quality_start_limiting = 3.

        # Asymptotic limit of soft-capped raw quality score.
        # dft value 1% percentile of quality in default parameters for all items ranked
        self.quality_limit = 13.6

        # If an item is organically allocated, discount for quality ENUE residual above this position.
        self.organic_alloc_qual_discount = 1.

        # negative (platform) experience penalty per repeated item for the same advertiser
        # Added to each duplicate
        self.same_promoter_additive = -1.
        # Multiple on pre-additive-duplicate-penalty Expected Negative User Experience
        self.same_promoter_multi = 1.5

        # conversion from units of Negative Expected User Experience to USD for comparison with bids
        self.exchange_rate = 1.

        # parameter to discount residual ENUE in higher positions if item is already allocated
        self.organic_discount = 1.

        # Minimum promoter bid required for delivery (and lowest possible price)
        # set by a function of promoter distribution, not position
        # units of cost per single insertion
        self.reserve_price = 0.1  # CPM, USD price per 1000 impressions

        self.utility_floor = 0

        # filter allocation to heuristic of top k
        # set to -1 to disable, 0 to only select organically allocated items
        self.pre_allocate_top_k = 1000
