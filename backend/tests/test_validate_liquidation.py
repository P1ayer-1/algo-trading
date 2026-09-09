"""The liquidation-model check, against a fixture of a real position.

The numbers below are a real BloFin demo position read on 2026-09-09, kept
verbatim. That matters more than a synthetic case would: the whole point of
this check is agreement with an exchange whose exact conventions - mark price
vs average price, contracts vs base units, cross vs isolated - are the things
most likely to be got wrong, and a fixture invented to match our own formula
would agree with it by construction.
"""

from decimal import Decimal

import pytest

from analysis.validate_liquidation import check_position, decimal_or_none

# Verbatim from GET /api/v1/account/positions, demo, 2026-09-09.
REAL_POSITION = {
    "instId": "SOL-USDT",
    "instType": "SWAP",
    "marginMode": "cross",
    "positionSide": "net",
    "positions": "-312.5",
    "averagePrice": "103.230000000000000000",
    "markPrice": "103.202468438",
    "liquidationPrice": "253.744121340977570628",
    "maintenanceMargin": "161.253856934375",
    "initialMargin": "429.936150817708333333",
    "leverage": "75",
}
REAL_BALANCE = Decimal("47477.731420959453157422")


def check(position=None, balance=REAL_BALANCE, contract_value=Decimal(1)):
    return check_position(position or dict(REAL_POSITION),
                          account_balance=balance,
                          contract_value=contract_value)


def test_the_model_agrees_with_the_exchange_on_a_real_position():
    """Roadmap step 1's outstanding item, as a regression test."""
    result = check()
    assert result["comparison"]["ok"], result["comparison"]
    assert result["comparison"]["relativeError"] < Decimal("0.001")


def test_the_maintenance_margin_rate_is_read_not_assumed():
    """MMR is tiered by size, so the base rate is the wrong number to use.

    Read off this position it is exactly 0.5%, which happens to match
    risk.py's default - but it is a measurement here, and a bigger position
    would land in a different tier.
    """
    result = check()
    assert result["mmr"] == pytest.approx(Decimal("0.005"), abs=1e-9)


def test_a_short_is_detected_from_the_sign_of_the_position():
    assert check()["side"] == "SHORT"

    long_position = dict(REAL_POSITION, positions="312.5")
    assert check(long_position)["side"] == "LONG"


def test_contracts_are_converted_to_base_units():
    """Positions come in CONTRACTS and risk.py works in base units.

    On an instrument whose contract is not one unit, skipping this produces a
    plausible-looking wrong answer rather than an obvious failure.
    """
    tenth = check(contract_value=Decimal("0.1"))
    assert tenth["quantityBase"] == Decimal("31.25")
    assert tenth["quantityBase"] != check()["quantityBase"]


def test_cross_margin_is_backed_by_the_whole_account():
    """Which is what cross margin means, and why the balance is the input.

    Halving the account balance must move a short's liquidation price closer
    to entry: less equity behind the position, less room before it goes.
    """
    full = check()
    thin = check(balance=REAL_BALANCE / 2)
    assert thin["estimate"] < full["estimate"]


def test_isolated_margin_uses_the_position_margin_not_the_account():
    isolated = dict(REAL_POSITION, marginMode="isolated",
                    margin="500.0")
    result = check(isolated)
    # 500 backing rather than 47,477 puts liquidation far closer to entry.
    assert result["estimate"] < Decimal("110")


def test_a_position_missing_a_field_is_reported_not_guessed():
    for field in ("averagePrice", "liquidationPrice", "maintenanceMargin",
                  "leverage", "markPrice"):
        broken = dict(REAL_POSITION)
        del broken[field]
        assert "error" in check(broken), f"{field} should be required"


def test_a_zero_size_position_is_not_compared():
    assert "error" in check(dict(REAL_POSITION, positions="0"))


def test_the_residual_points_the_unsafe_way_for_a_short():
    """The model must be known to be optimistic, not silently so.

    BloFin carries a liquidation fee rate in the denominator that the clean
    derivation omits, so our estimate sits slightly FURTHER from entry than
    the exchange's. For a short that means a higher price - liquidation looks
    further away than it is, which is the direction worth knowing about.
    """
    result = check()
    assert result["estimate"] > result["reported"]
    assert result["impliedFeeRate"] > 0
    # A fee rate, not a bug: single basis points, not tens.
    assert result["impliedFeeRate"] < Decimal("0.001")


def test_decimal_or_none_survives_junk():
    assert decimal_or_none("1.5") == Decimal("1.5")
    assert decimal_or_none(None) is None
    assert decimal_or_none("") is None
    assert decimal_or_none("not a number") is None
