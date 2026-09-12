"""The cross-venue pair: the sign, the alignment, and what the control is for.

The sign is first because it is the failure that actually happened while this
was being written. Taking `+side * difference` instead of `-side * difference`
puts the book on the LOSING side of a spread it identified correctly, and the
result reads as a strategy that does not work rather than as a bug - which is
the most expensive kind of wrong, because it gets believed and filed away.
"""

import numpy as np
import pytest

from analysis.factor_panel import Panel
from analysis.funding_dispersion import Aligned, align, backtest


def aligned(funding_a, funding_b, close_a=None, close_b=None, volume=1e9):
    funding_a = np.asarray(funding_a, dtype=float)
    funding_b = np.asarray(funding_b, dtype=float)
    shape = funding_a.shape
    ones = np.ones(shape)
    return Aligned(
        dates=["2026-01-{:02d}".format(d + 1) for d in range(shape[0])],
        symbols=["S{}".format(i) for i in range(shape[1])],
        funding_a=funding_a, funding_b=funding_b,
        close_a=(np.asarray(close_a, dtype=float) if close_a is not None
                 else ones * 100.0),
        close_b=(np.asarray(close_b, dtype=float) if close_b is not None
                 else ones * 100.0),
        volume=ones * volume, name_a="A", name_b="B")


def panel(close, funding, symbols, dates):
    close = np.asarray(close, dtype=float)
    ones = np.ones_like(close)
    return Panel(dates=list(dates), symbols=list(symbols), close=close,
                 volume=ones * 1e9, funding=np.asarray(funding, dtype=float),
                 rv=ones, high=close, low=close, taker=ones * 0.5,
                 complete=np.ones_like(close, dtype=bool))


# ---------------------------------------------------------------------------
# The sign
# ---------------------------------------------------------------------------


def test_shorting_the_dearer_venue_collects_the_difference():
    """Venue A's longs pay 5 bps a day more, so short A / long B EARNS.

    A short receives the funding its longs pay, so a position that is short A
    and long B collects `+(funding_A - funding_B)`. Getting this backwards
    makes a real and persistent spread look like a losing trade.
    """
    n_dates = 40
    funding_a = np.full((n_dates, 4), 6.0)
    funding_b = np.full((n_dates, 4), 1.0)
    result = backtest(aligned(funding_a, funding_b), hold_days=7, top=4,
                      cost_bps=0.0, min_volume=0.0, start=7, lookback=7)
    assert len(result.net) >= 3
    # 5 bps a day for 7 days is 35 bps on one unit of notional a side; the
    # report is per unit of GROSS notional, and the pair is 2 gross.
    assert result.funding.mean() == pytest.approx(17.5, rel=1e-6)
    assert result.net.mean() > 0


def test_the_direction_flips_with_the_sign_of_the_difference():
    """When venue B is the dearer one the book must reverse, not lose.

    The strategy is symmetric by construction; a version that only worked when
    one particular venue happened to be expensive would be a bet on that venue,
    which is a different and much weaker claim.
    """
    n_dates = 40
    result = backtest(aligned(np.full((n_dates, 4), 1.0), np.full((n_dates, 4), 6.0)),
                      hold_days=7, top=4, cost_bps=0.0, min_volume=0.0,
                      start=7, lookback=7)
    assert result.funding.mean() == pytest.approx(17.5, rel=1e-6)


# ---------------------------------------------------------------------------
# The price leg
# ---------------------------------------------------------------------------


def test_the_coins_own_move_cancels_and_only_divergence_survives():
    """Both legs are the same underlying, so a 50% rally is not a P&L.

    This is the whole reason the trade is worth doing: the price risk that
    makes the cross-sectional carry book swing 300 bps a week is cancelled
    here by construction. A harness that credited the coin's move would report
    an enormous and entirely fictional variance.
    """
    n_dates = 40
    close = np.ones((n_dates, 4)) * 100.0
    close[10:] = 150.0                       # both venues, same move
    result = backtest(aligned(np.full((n_dates, 4), 6.0), np.full((n_dates, 4), 1.0),
                              close_a=close, close_b=close),
                      hold_days=7, top=4, cost_bps=0.0, min_volume=0.0,
                      start=7, lookback=7)
    assert np.allclose(result.divergence, 0.0)


def test_a_venue_that_diverges_against_the_short_leg_costs_money():
    """Venue A running away from venue B is the real risk of the pair.

    The book is short A, so A outperforming B is a loss, and it has to show up
    in `divergence` rather than being assumed away. 100 bps of divergence on a
    2-gross pair is 50 bps per unit of gross notional.
    """
    n_dates = 40
    close_a = np.ones((n_dates, 4)) * 100.0
    close_b = np.ones((n_dates, 4)) * 100.0
    close_a[8:] = 100.0 * np.exp(0.01)       # A gains 100 bps on B, once
    result = backtest(aligned(np.full((n_dates, 4), 6.0), np.full((n_dates, 4), 1.0),
                              close_a=close_a, close_b=close_b),
                      hold_days=7, top=4, cost_bps=0.0, min_volume=0.0,
                      start=7, lookback=7)
    hit = result.divergence[result.divergence != 0]
    assert len(hit) == 1
    assert hit[0] == pytest.approx(-50.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Cost and sampling
# ---------------------------------------------------------------------------


def test_cost_is_four_taker_legs_per_round_trip():
    """In and out, on two venues. Charging two legs would halve the bar."""
    n_dates = 40
    flat = aligned(np.zeros((n_dates, 4)), np.zeros((n_dates, 4)))
    result = backtest(flat, hold_days=7, top=4, cost_bps=5.0, min_volume=0.0,
                      start=7, lookback=7)
    assert np.allclose(result.net, -10.0)


def test_holding_periods_do_not_overlap():
    n_dates = 60
    result = backtest(aligned(np.full((n_dates, 4), 3.0), np.zeros((n_dates, 4))),
                      hold_days=7, top=4, cost_bps=0.0, min_volume=0.0,
                      start=7, lookback=7)
    assert result.dates == sorted(set(result.dates))
    assert len(result.net) <= (n_dates - 7) // 7 + 1


def test_illiquid_coins_are_excluded_on_the_smaller_venue():
    """The pair can only be as big as its thinner leg, so the volume filter
    takes the MINIMUM of the two venues rather than either one's own.

    The two illiquid coins are given a large venue divergence, so including
    them would move `divergence` away from zero. Asserting only that the
    backtest still produced periods would pass whether or not the filter did
    anything.
    """
    n_dates = 40
    close_a = np.ones((n_dates, 6)) * 100.0
    close_a[:, :2] = 100.0 * np.exp(np.linspace(0, 0.5, n_dates))[:, None]
    data = aligned(np.full((n_dates, 6), 6.0), np.full((n_dates, 6), 1.0),
                   close_a=close_a)
    data.volume[:, :2] = 1.0             # the two divergent ones are illiquid

    filtered = backtest(data, hold_days=7, top=6, cost_bps=0.0,
                        min_volume=1e6, start=7, lookback=7)
    unfiltered = backtest(data, hold_days=7, top=6, cost_bps=0.0,
                          min_volume=0.0, start=7, lookback=7)
    assert len(filtered.net) >= 3
    assert np.allclose(filtered.divergence, 0.0)
    assert not np.allclose(unfiltered.divergence, 0.0)


def test_selection_and_its_control_hold_the_same_number_of_pairs():
    """The control picks coins at random and changes nothing else, so a
    difference between them is selection skill and not book size."""
    rng = np.random.default_rng(0)
    n_dates = 120
    data = aligned(rng.normal(3.0, 2.0, (n_dates, 8)), np.zeros((n_dates, 8)))
    real = backtest(data, hold_days=7, top=4, cost_bps=0.0, min_volume=0.0,
                    start=7, lookback=7)
    control = backtest(data, hold_days=7, top=4, cost_bps=0.0, min_volume=0.0,
                       start=7, lookback=7, shuffle_seed=1)
    assert len(real.net) == len(control.net)
    assert real.positions == control.positions


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_align_intersects_dates_and_symbols():
    """Two panels covering different windows and universes must be compared
    only where both speak. A left join would silently read one venue's funding
    against the other's absence."""
    dates_a = ["2026-01-01", "2026-01-02", "2026-01-03"]
    dates_b = ["2026-01-02", "2026-01-03", "2026-01-04"]
    a = panel(np.ones((3, 2)) * 10.0, np.ones((3, 2)) * 5.0, ["X", "Y"], dates_a)
    b = panel(np.ones((3, 2)) * 10.0, np.ones((3, 2)) * 2.0, ["Y", "Z"], dates_b)
    data = align(a, b, name_a="A", name_b="B")
    assert data.dates == ["2026-01-02", "2026-01-03"]
    assert data.symbols == ["Y"]
    assert np.allclose(data.difference, 3.0)
