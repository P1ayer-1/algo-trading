"""The cross-sectional harness: the five ways a factor backtest flatters itself.

Each test names the specific overstatement it prevents. Overlapping holds
inflate the sample; a label measured against zero pays the drift; funding
counted from the wrong end hands the book a free day of carry; a universe
filtered on the whole sample selects the survivors; and a cost charged on
positions rather than on turnover under-charges a book that keeps its names.
"""

import numpy as np
import pytest

from analysis.factor_panel import (
    Panel,
    block_bootstrap,
    build_features,
    holding_return,
    run_factor,
    tradeable,
    _weights,
)


def panel(close, *, funding=None, volume=None, complete=None, symbols=None):
    """A `(dates, symbols)` panel from a close matrix, everything else filled."""
    close = np.asarray(close, dtype=float)
    n_dates, n_symbols = close.shape
    ones = np.ones_like(close)
    return Panel(
        dates=["2026-01-{:02d}".format(d + 1) for d in range(n_dates)],
        symbols=symbols or ["S{}".format(i) for i in range(n_symbols)],
        close=close,
        volume=np.asarray(volume, dtype=float) if volume is not None else ones * 1e9,
        funding=(np.asarray(funding, dtype=float) if funding is not None
                 else np.zeros_like(close)),
        rv=ones * 100.0,
        high=close * 1.01,
        low=close * 0.99,
        taker=ones * 0.5,
        complete=(np.asarray(complete, dtype=bool) if complete is not None
                  else np.ones_like(close, dtype=bool)),
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_holding_return_charges_funding_from_the_day_after_entry():
    """Day `entry`'s funding accrued BEFORE the position existed.

    The position is opened at the close of day `entry`, by which time that
    day's carry has already settled. Including it is a free day of funding
    handed to the book in whichever direction the factor happened to lean, and
    at a 7-day hold that is a seventh of the whole carry signal.
    """
    prices = np.array([[100.0], [100.0], [100.0], [100.0]])
    funding = np.array([[99.0], [1.0], [2.0], [4.0]])
    result = holding_return(panel(prices, funding=funding), 0, 3)
    assert result[0] == pytest.approx(-(1.0 + 2.0 + 4.0))


def test_holding_return_is_price_minus_funding():
    """A long perp earns the price move and PAYS positive funding.

    Sign errors here are invisible in a symmetric book and decisive in a carry
    factor, whose entire return is the funding leg.
    """
    prices = np.array([[100.0], [100.0], [110.0]])
    funding = np.array([[0.0], [5.0], [5.0]])
    result = holding_return(panel(prices, funding=funding), 0, 2)
    expected = np.log(110.0 / 100.0) * 10_000.0 - 10.0
    assert result[0] == pytest.approx(expected)


def test_a_market_wide_move_earns_the_book_nothing():
    """Every symbol up 10% is drift, not skill.

    Step 6 found this exact failure: measuring a decile's return against zero
    rather than against the sample's own drift scored "short everything" as an
    edge in a falling window. A dollar-neutral book cannot capture drift, so
    the harness must report zero here whatever the factor says.
    """
    close = np.ones((40, 10)) * 100.0
    close[20:] = 110.0
    scores = np.tile(np.arange(10, dtype=float), (40, 1))
    eligible = np.ones((40, 10), dtype=bool)
    result = run_factor(panel(close), scores, eligible, hold_days=5,
                        top_frac=0.3, cost_bps=0.0, start=0)
    assert result.periods
    assert max(abs(p.gross_bps) for p in result.periods) < 1e-9


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_holding_periods_do_not_overlap():
    """The count of rebalances is the effective N only if they are disjoint.

    A 7-day hold rebalanced daily gives seven times the rows and none of the
    independence, and every t-statistic computed on it is off by a factor of
    2.6. The harness offers no overlap correction because it does not overlap.
    """
    close = np.cumprod(1 + np.zeros((60, 8)), axis=0) * 100.0
    scores = np.tile(np.arange(8, dtype=float), (60, 1))
    result = run_factor(panel(close), scores, np.ones((60, 8), dtype=bool),
                        hold_days=7, top_frac=0.25, cost_bps=0.0, start=0)
    spans = [(p.entry, p.exit_index) for p in result.periods]
    assert spans == sorted(spans)
    for (_, previous_exit), (next_entry, _) in zip(spans, spans[1:]):
        assert next_entry >= previous_exit, "holding periods overlap"


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def test_universe_is_point_in_time_on_liquidity():
    """A coin that gets liquid in year three was not tradeable in year one.

    Filtering on full-sample volume is the version of survivorship this file
    can actually avoid, and the one that most flatters a cross-sectional book:
    it fills the early years with the names that later got big.
    """
    n_dates = 80
    volume = np.ones((n_dates, 2)) * 1e9
    volume[:, 1] = 1.0                       # illiquid...
    volume[60:, 1] = 1e9                     # ...until day 60
    eligible = tradeable(panel(np.ones((n_dates, 2)) * 100.0, volume=volume),
                         min_history=30, min_volume=1e6, volume_window=30)
    assert eligible[50, 1] == False          # noqa: E712 - explicit about the bit
    assert eligible[50, 0] == True           # noqa: E712
    assert eligible[79, 1] == True           # noqa: E712


def test_universe_requires_an_unbroken_run_of_complete_days():
    """A gap resets the history counter rather than being counted through.

    A symbol with 90 rows of which 40 are stubs has not got 90 days of history,
    and the return across the stub spans an unknown number of days.
    """
    n_dates = 80
    complete = np.ones((n_dates, 1), dtype=bool)
    complete[40, 0] = False
    eligible = tradeable(panel(np.ones((n_dates, 1)) * 100.0, complete=complete),
                         min_history=30, min_volume=0.0, volume_window=30)
    assert eligible[39, 0] == True           # noqa: E712
    assert eligible[60, 0] == False          # noqa: E712 - only 20 days since
    assert eligible[75, 0] == True           # noqa: E712


# ---------------------------------------------------------------------------
# Weights and cost
# ---------------------------------------------------------------------------


def test_weights_are_dollar_neutral_and_gross_two():
    """One dollar long and one dollar short, so `sum|w|` is 2.

    Every return in the report is divided by that 2, which is what makes a
    figure here comparable with the per-trade bps of the earlier steps.
    """
    scores = np.arange(10, dtype=float)
    weights, n_long, n_short = _weights(scores, np.ones(10, dtype=bool), 0.3)
    assert n_long == n_short == 3
    assert weights.sum() == pytest.approx(0.0)
    assert np.abs(weights).sum() == pytest.approx(2.0)
    assert weights[9] > 0 and weights[0] < 0


def test_ineligible_symbols_are_never_held():
    scores = np.arange(10, dtype=float)
    eligible = np.ones(10, dtype=bool)
    eligible[9] = False
    weights, _, _ = _weights(scores, eligible, 0.3)
    assert weights[9] == 0.0
    assert weights[8] > 0


def test_cost_is_charged_on_turnover_not_on_position():
    """A book that keeps its names pays nothing to keep them.

    Charging a round trip every rebalance regardless of turnover is the
    pessimistic error, and it hides the real one: a factor whose ranking churns
    pays far more than a stable one, and the whole point of a cost model is to
    let the two be compared. A constant score never trades after the first
    rebalance, so only that first one may be charged.
    """
    close = np.ones((40, 10)) * 100.0
    scores = np.tile(np.arange(10, dtype=float), (40, 1))
    result = run_factor(panel(close), scores, np.ones((40, 10), dtype=bool),
                        hold_days=5, top_frac=0.3, cost_bps=10.0, start=0)
    assert result.periods[0].turnover == pytest.approx(2.0)
    assert all(p.turnover == 0.0 for p in result.periods[1:])
    assert all(p.net_bps == 0.0 for p in result.periods[1:])


def test_a_name_that_stops_printing_mid_hold_is_dropped_not_zeroed():
    """A delisting is not a return of zero.

    Treating a missing exit price as flat is the single most flattering bug
    available to a long/short book, because the names that vanish are the ones
    that went to zero. The position is dropped and the rest renormalised, which
    is optimistic in a different direction and at least says so.
    """
    close = np.ones((20, 8)) * 100.0
    close[10:, 3] = np.nan
    scores = np.tile(np.arange(8, dtype=float), (20, 1))
    result = run_factor(panel(close), scores, np.ones((20, 8), dtype=bool),
                        hold_days=5, top_frac=0.25, cost_bps=0.0, start=0)
    for period in result.periods:
        if period.entry >= 5:
            assert period.weights[3] == 0.0


# ---------------------------------------------------------------------------
# Beta, which the shuffled control cannot catch
# ---------------------------------------------------------------------------


def test_a_book_that_is_only_carrying_market_has_no_alpha():
    """Dollar-neutral is not beta-neutral, and the control cannot see the
    difference.

    Sorting on anything vol-adjacent puts high-beta names on one side, so in a
    drifting market the book earns the drift times its tilt. A shuffled control
    has no systematic tilt, so it does NOT reproduce that return and the factor
    appears to beat it. Here the long side moves at twice the market and the
    short side at zero, which is pure beta: the net is large and the alpha must
    be ~0.
    """
    rng = np.random.default_rng(11)
    n_dates, n_symbols = 220, 10
    market_step = rng.normal(0.004, 0.02, n_dates)
    beta = np.linspace(0.0, 2.0, n_symbols)
    log_price = np.cumsum(np.outer(market_step, beta), axis=0)
    close = 100.0 * np.exp(log_price)
    scores = np.tile(beta, (n_dates, 1))          # rank by beta, constant

    result = run_factor(panel(close), scores, np.ones((n_dates, n_symbols), dtype=bool),
                        hold_days=7, top_frac=0.3, cost_bps=0.0, start=0)
    summary = result.summary(365.0 / 7.0)
    assert summary["beta"] > 0.5, "the harness did not detect the market tilt"
    assert abs(summary["alpha_bps"]) < abs(summary["net_bps"]) / 3.0

    control = run_factor(panel(close), scores,
                         np.ones((n_dates, n_symbols), dtype=bool), hold_days=7,
                         top_frac=0.3, cost_bps=0.0, start=0, shuffle_seed=1)
    assert abs(control.summary(365.0 / 7.0)["beta"]) < abs(summary["beta"]) / 2.0, \
        "the control reproduced the tilt, so it would have caught this by itself"


def test_by_year_splits_the_periods_without_losing_any():
    close = np.ones((120, 8)) * 100.0
    scores = np.tile(np.arange(8, dtype=float), (120, 1))
    result = run_factor(panel(close), scores, np.ones((120, 8), dtype=bool),
                        hold_days=7, top_frac=0.25, cost_bps=0.0, start=0)
    table = result.by_year()
    assert sum(count for count, _ in table.values()) == len(result.periods)


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------


def test_a_perfect_factor_earns_and_its_shuffled_control_does_not():
    """The control keeps the universe, the count and the cost, and breaks only
    the pairing between a symbol and its score.

    If a shuffled factor scored like the real one, the harness would be
    measuring the universe rather than the ranking - which is precisely what
    step 8's control caught when the top-decile metric turned out to be biased
    upward for anything correlated with volatility.
    """
    rng = np.random.default_rng(0)
    n_dates, n_symbols = 210, 12
    scores = rng.normal(size=(n_dates, n_symbols))
    close = np.ones((n_dates, n_symbols)) * 100.0
    for entry in range(0, n_dates - 7, 7):
        # Next period's return is the score, so the ranking is perfect.
        close[entry + 1:entry + 8] = close[entry] * np.exp(scores[entry] * 0.01)

    eligible = np.ones((n_dates, n_symbols), dtype=bool)
    real = run_factor(panel(close), scores, eligible, hold_days=7,
                      top_frac=0.25, cost_bps=0.0, start=0)
    control = run_factor(panel(close), scores, eligible, hold_days=7,
                         top_frac=0.25, cost_bps=0.0, start=0, shuffle_seed=1)
    assert real.net.mean() > 100.0
    assert abs(control.net.mean()) < real.net.mean() / 4.0
    assert real.mean_ic > 0.9


def test_control_holds_the_same_number_of_positions():
    """Same book size, same turnover distribution, same cost - or the control
    is measuring position count rather than ranking skill."""
    rng = np.random.default_rng(3)
    close = np.cumprod(1 + rng.normal(0, 0.01, (120, 10)), axis=0) * 100.0
    scores = rng.normal(size=(120, 10))
    eligible = np.ones((120, 10), dtype=bool)
    real = run_factor(panel(close), scores, eligible, hold_days=7,
                      top_frac=0.3, cost_bps=5.0, start=0)
    control = run_factor(panel(close), scores, eligible, hold_days=7,
                         top_frac=0.3, cost_bps=5.0, start=0, shuffle_seed=2)
    assert len(real.periods) == len(control.periods)
    assert [p.n_long for p in real.periods] == [p.n_long for p in control.periods]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_carry_factor_is_negative_funding():
    """High carry score means "expect to outperform", and a perp whose longs
    are paying is one to be SHORT. Fixing the sign in the feature rather than
    at scoring time is what stops a factor being flipped after its result is
    seen."""
    close = np.ones((40, 2)) * 100.0
    funding = np.zeros((40, 2))
    funding[:, 0] = 3.0                      # longs pay here
    funding[:, 1] = -3.0                     # shorts pay here
    features = build_features(panel(close, funding=funding))
    assert features["carry_7"][39, 0] == pytest.approx(-3.0)
    assert features["carry_7"][39, 1] == pytest.approx(3.0)


def test_features_never_see_the_bar_they_are_stated_at_plus_one():
    """A feature closed at day d must not move when day d+1 changes.

    This is the leakage check that every previous step in this repo needed and
    one of them did not have: shifting a trailing window by a single row is
    invisible in the output and doubles an IC.
    """
    rng = np.random.default_rng(5)
    close = np.cumprod(1 + rng.normal(0, 0.01, (150, 4)), axis=0) * 100.0
    funding = rng.normal(0, 2.0, (150, 4))
    base = build_features(panel(close, funding=funding))

    altered_close = close.copy()
    altered_funding = funding.copy()
    altered_close[100:] *= 3.0
    altered_funding[100:] += 50.0
    altered = build_features(panel(altered_close, funding=altered_funding))

    for name, values in base.items():
        row, other = values[99], altered[name][99]
        both = np.isfinite(row) & np.isfinite(other)
        assert np.allclose(row[both], other[both]), name + " saw the future"


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def test_block_bootstrap_interval_covers_the_mean_and_widens_with_noise():
    quiet = np.full(100, 5.0) + np.random.default_rng(1).normal(0, 0.1, 100)
    noisy = np.full(100, 5.0) + np.random.default_rng(1).normal(0, 10.0, 100)
    quiet_lo, quiet_hi = block_bootstrap(quiet)
    noisy_lo, noisy_hi = block_bootstrap(noisy)
    assert quiet_lo < 5.0 < quiet_hi
    assert (noisy_hi - noisy_lo) > (quiet_hi - quiet_lo) * 10


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_clustering_demeans_before_correlating():
    """Without demeaning, single linkage chains every crypto into one cluster.

    Measured on the real panel at 2024-12-25: a 0.75 threshold on raw returns
    put 61 of 64 names in ONE cluster and left three singletons, because
    everything correlates through its beta to the market and single linkage
    chains A-B-C through it. Weighting on that hands almost the whole book to
    whichever three names happened not to chain.

    Here every name is beta 1 to a common factor plus its own noise, so on raw
    returns they all chain and on residuals none of them should.
    """
    from analysis.factor_panel import correlation_clusters

    rng = np.random.default_rng(0)
    days, names = 400, 8
    market = rng.normal(0, 0.03, (days, 1))
    returns = market + rng.normal(0, 0.005, (days, names))
    assert len(set(correlation_clusters(returns, threshold=0.75))) == names


def test_names_that_move_together_beyond_their_beta_share_a_cluster():
    from analysis.factor_panel import correlation_clusters

    rng = np.random.default_rng(1)
    days, names = 400, 8
    market = rng.normal(0, 0.03, (days, 1))
    returns = market + rng.normal(0, 0.01, (days, names))
    shared = rng.normal(0, 0.05, (days, 1))          # a second, narrower factor
    returns[:, :3] += shared
    labels = correlation_clusters(returns, threshold=0.75)
    assert labels[0] == labels[1] == labels[2]
    assert len({labels[i] for i in range(3, names)} & {labels[0]}) == 0


def test_cluster_weighting_gives_a_cluster_one_clusters_worth_of_money():
    """Three names that move as one should carry the risk of one position."""
    from analysis.factor_panel import _weights

    scores = np.arange(8, dtype=float)
    clusters = np.array([0, 0, 0, 1, 2, 3, 4, 5])
    weights, _, _ = _weights(scores, np.ones(8, dtype=bool), 0.5, None, clusters)
    shorts = weights[:4]
    # Names 0,1,2 share a cluster and name 3 is alone, so the cluster and the
    # singleton each get half of the short side.
    assert abs(shorts[:3].sum()) == pytest.approx(abs(shorts[3]))
    assert np.abs(weights).sum() == pytest.approx(2.0)
