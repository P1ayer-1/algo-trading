"""Attention across scales: the gradients, and whether it can learn at all.

A hand-derived backward pass is worth exactly as much as its gradient check, so
that is the first test here. The second is the one that makes the first mean
something: a planted CROSS-SCALE interaction that a linear model cannot
represent, which this must beat ridge on - otherwise the architecture is
decoration and every result from it is a result about noise.
"""

import numpy as np
import pytest

from analysis.attention_model import ScaleAttention
from analysis.range_harness import Ridge


def tiny_model(**kwargs):
    return ScaleAttention(n_scales=3, d_model=4, d_hidden=3, seed=1, **kwargs)


def test_every_gradient_matches_finite_differences():
    """The whole model rests on this. Central differences, one parameter at a
    time, against the analytic backward pass."""
    model = tiny_model(l2=0.01)
    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(7, 3, 5))
    y = rng.normal(size=7)
    params = model._build(5)

    _, grads = model.loss_and_grads(tokens, y, params)
    step = 1e-6
    for name, value in params.items():
        flat = value.ravel()
        for index in range(len(flat)):
            original = flat[index]
            flat[index] = original + step
            up, _ = model.loss_and_grads(tokens, y, params)
            flat[index] = original - step
            down, _ = model.loss_and_grads(tokens, y, params)
            flat[index] = original
            numeric = (up - down) / (2 * step)
            assert grads[name].ravel()[index] == pytest.approx(numeric, abs=2e-6), (
                f"{name}[{index}]")


def test_attention_weights_are_a_distribution_over_scales():
    model = tiny_model()
    rng = np.random.default_rng(2)
    tokens = rng.normal(size=(11, 3, 5))
    _, cache = model.forward(tokens, model._build(5))
    assert cache["A"].shape == (11, 3, 3)
    assert cache["A"].sum(axis=-1) == pytest.approx(np.ones((11, 3)))
    assert np.all(cache["A"] >= 0)


def test_features_are_split_into_one_token_per_scale():
    model = tiny_model()
    flat = np.arange(2 * 3 * 5, dtype=float).reshape(2, 15)
    tokens = model._tokens(flat)
    assert tokens.shape == (2, 3, 5)
    assert tokens[0, 0] == pytest.approx(flat[0, :5])
    assert tokens[0, 2] == pytest.approx(flat[0, 10:])


def test_a_feature_count_that_does_not_divide_by_the_scales_is_refused():
    with pytest.raises(ValueError):
        tiny_model()._tokens(np.zeros((4, 11)))


def test_it_beats_ridge_on_an_interaction_between_two_scales():
    """y depends on the PRODUCT of a feature in scale 0 and one in scale 2, so
    no linear combination of the 15 columns can express it. If attention cannot
    win here, it cannot be said to be using the cross-scale structure."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, 3, 5))
    y = X[:, 0, 0] * X[:, 2, 1]
    flat = X.reshape(len(X), -1)
    train, test = slice(0, 3000), slice(3000, None)

    model = ScaleAttention(n_scales=3, d_model=16, d_hidden=16, epochs=40,
                           learning_rate=0.02, l2=1e-5, seed=1)
    model.fit(flat[train], y[train])
    attention_error = np.mean((model.predict(flat[test]) - y[test]) ** 2)

    ridge = Ridge(l2=1.0)
    ridge.fit(flat[train], y[train])
    ridge_error = np.mean((ridge.predict(flat[test]) - y[test]) ** 2)

    assert attention_error < 0.7 * ridge_error


def test_it_recovers_a_plain_linear_signal_too():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3000, 3, 5))
    y = 1.5 * X[:, 1, 2]
    flat = X.reshape(len(X), -1)
    model = ScaleAttention(n_scales=3, d_model=16, d_hidden=16, epochs=30,
                           learning_rate=0.02, seed=2)
    model.fit(flat[:2200], y[:2200])
    predicted = model.predict(flat[2200:])
    assert np.corrcoef(predicted, y[2200:])[0, 1] > 0.9


def test_predicting_before_fitting_is_refused():
    with pytest.raises(RuntimeError):
        tiny_model().predict(np.zeros((2, 15)))


def test_the_same_seed_gives_the_same_model():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 15))
    y = rng.normal(size=400)
    first = ScaleAttention(n_scales=3, epochs=2, seed=7)
    second = ScaleAttention(n_scales=3, epochs=2, seed=7)
    first.fit(X, y)
    second.fit(X, y)
    assert first.predict(X) == pytest.approx(second.predict(X))


def test_early_stopping_keeps_the_best_epoch_not_the_last():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(600, 15))
    y = rng.normal(size=600)              # nothing to learn, so it must stop early
    model = ScaleAttention(n_scales=3, epochs=30, patience=2, seed=3)
    model.fit(X, y)
    assert len(model.history) < 30
    assert min(model.history) == pytest.approx(model.history[-model.patience - 1],
                                               rel=1e-9)


def test_the_attention_map_is_square_over_scales():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 15))
    model = ScaleAttention(n_scales=3, epochs=2, seed=1)
    model.fit(X, rng.normal(size=300))
    table = model.attention_map(X)
    assert table.shape == (3, 3)
    assert table.sum(axis=1) == pytest.approx(np.ones(3))
