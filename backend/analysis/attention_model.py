"""Attention across timeframes, in numpy, scored on the same bar as everything else.

One token per scale, self-attention across the tokens, a small head on the
pooled result. The point is that the 4h token can read the 7d token before
anything is predicted - which is the part a ridge over concatenated features
cannot do, and the part step 9o could not rule out by argument.

Why hand-written rather than a framework
----------------------------------------
The model is ~1,300 parameters over four tokens. Torch would add a multi-GB
dependency to an analysis layer that needs numpy alone, for matrices this size,
so the gradients are derived here and CHECKED against finite differences in
`tests/test_attention_model.py`. A hand-derived backward pass is worth exactly
as much as its gradient check, which is why that test is the first one.

Shape, end to end, for a batch of B rows over S scales with F features each:

    X      (B, S, F)   standardised features, one token per scale
    H      (B, S, d)   X @ W_in + b_in + E, where E is a learned per-scale
                       embedding - without it the tokens are interchangeable
                       and the attention cannot tell 4h from 7d
    A      (B, S, S)   softmax(Q K^T / sqrt(d)), which scale reads which
    U      (B, S, d)   H + A V, a residual so the head still sees each token
    p      (B, S*d)    tokens CONCATENATED, not averaged: the input projection
                       is shared across scales, so mean-pooling would leave the
                       head reading only the average token and a signal living
                       in one scale would be diluted away (measured: it could
                       not recover even a plain linear target)
    yhat   (B,)        w2 . tanh(W1 p + b1) + b2

`fit(X, y)` / `predict(X)` match the harness Model protocol structurally, so
`range_harness.evaluate` scores this exactly as it scores ridge: purged split,
shuffled control, effective N, and the IC converted to bps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


def _init(rng: np.random.Generator, *shape: int, scale: float = 1.0) -> np.ndarray:
    fan_in = shape[0] if shape else 1
    return rng.normal(0.0, scale / np.sqrt(fan_in), shape)


@dataclass
class ScaleAttention:
    """Self-attention over per-scale tokens. Structurally a harness Model."""

    n_scales: int
    d_model: int = 16
    d_hidden: int = 16
    learning_rate: float = 0.01
    batch_size: int = 1024
    epochs: int = 12
    l2: float = 1e-4
    patience: int = 3
    validation_fraction: float = 0.1
    seed: int = 0
    name: str = "attention"
    params: Dict[str, np.ndarray] = field(default_factory=dict)
    y_mean: float = 0.0
    y_std: float = 1.0
    history: list = field(default_factory=list)

    # -- plumbing ----------------------------------------------------------

    def _tokens(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return X
        if X.shape[1] % self.n_scales:
            raise ValueError(
                f"{X.shape[1]} features do not divide into {self.n_scales} scales")
        return X.reshape(len(X), self.n_scales, X.shape[1] // self.n_scales)

    def _build(self, n_features: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        d, hidden = self.d_model, self.d_hidden
        return {
            "W_in": _init(rng, n_features, d),
            "b_in": np.zeros(d),
            "E": _init(rng, self.n_scales, d, scale=0.1),
            "W_q": _init(rng, d, d),
            "W_k": _init(rng, d, d),
            "W_v": _init(rng, d, d),
            "W1": _init(rng, self.n_scales * d, hidden),
            "b1": np.zeros(hidden),
            "w2": _init(rng, hidden, 1).ravel(),
            "b2": np.zeros(1),
        }

    # -- forward and backward ---------------------------------------------

    def forward(self, tokens: np.ndarray, params: Dict[str, np.ndarray]):
        H = np.einsum("bsf,fd->bsd", tokens, params["W_in"]) + params["b_in"] + params["E"]
        Q = np.einsum("bsd,de->bse", H, params["W_q"])
        K = np.einsum("bsd,de->bse", H, params["W_k"])
        V = np.einsum("bsd,de->bse", H, params["W_v"])
        scores = np.einsum("bse,bte->bst", Q, K) / np.sqrt(self.d_model)
        shifted = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(shifted)
        A = weights / weights.sum(axis=-1, keepdims=True)
        Z = np.einsum("bst,btd->bsd", A, V)
        U = H + Z
        pooled = U.reshape(len(U), -1)
        pre = pooled @ params["W1"] + params["b1"]
        hidden = np.tanh(pre)
        out = hidden @ params["w2"] + params["b2"][0]
        cache = dict(tokens=tokens, H=H, Q=Q, K=K, V=V, A=A, pooled=pooled,
                     hidden=hidden)
        return out, cache

    def backward(self, cache, params: Dict[str, np.ndarray],
                 dout: np.ndarray) -> Dict[str, np.ndarray]:
        H, Q, K, V, A = cache["H"], cache["Q"], cache["K"], cache["V"], cache["A"]
        pooled, hidden, tokens = cache["pooled"], cache["hidden"], cache["tokens"]
        scales = H.shape[1]

        grads = {name: np.zeros_like(value) for name, value in params.items()}
        grads["w2"] = hidden.T @ dout
        grads["b2"] = np.array([dout.sum()])

        d_hidden = np.outer(dout, params["w2"])
        d_pre = d_hidden * (1.0 - hidden ** 2)
        grads["W1"] = pooled.T @ d_pre
        grads["b1"] = d_pre.sum(axis=0)

        d_pooled = d_pre @ params["W1"].T
        dU = d_pooled.reshape(H.shape)
        dH = dU.copy()                       # the residual path
        dZ = dU

        dA = np.einsum("bsd,btd->bst", dZ, V)
        dV = np.einsum("bst,bsd->btd", A, dZ)
        # softmax backward, row by row over the last axis
        dScores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dScores /= np.sqrt(self.d_model)
        dQ = np.einsum("bst,btd->bsd", dScores, K)
        dK = np.einsum("bst,bsd->btd", dScores, Q)

        grads["W_q"] = np.einsum("bsd,bse->de", H, dQ)
        grads["W_k"] = np.einsum("bsd,bse->de", H, dK)
        grads["W_v"] = np.einsum("bsd,bse->de", H, dV)

        dH = dH + np.einsum("bse,de->bsd", dQ, params["W_q"])
        dH = dH + np.einsum("bse,de->bsd", dK, params["W_k"])
        dH = dH + np.einsum("bse,de->bsd", dV, params["W_v"])

        grads["W_in"] = np.einsum("bsf,bsd->fd", tokens, dH)
        grads["b_in"] = dH.sum(axis=(0, 1))
        grads["E"] = dH.sum(axis=0)
        return grads

    def loss_and_grads(self, tokens: np.ndarray, y: np.ndarray,
                       params: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:
        out, cache = self.forward(tokens, params)
        residual = out - y
        loss = float(np.mean(residual ** 2))
        dout = 2.0 * residual / len(y)
        grads = self.backward(cache, params, dout)
        if self.l2:
            for name in ("W_in", "W_q", "W_k", "W_v", "W1", "w2"):
                loss += self.l2 * float(np.sum(params[name] ** 2))
                grads[name] = grads[name] + 2.0 * self.l2 * params[name]
        return loss, grads

    # -- the Model protocol ------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        tokens = self._tokens(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        self.y_mean, self.y_std = float(y.mean()), float(y.std())
        if self.y_std == 0:
            self.y_std = 1.0
        target = (y - self.y_mean) / self.y_std

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(tokens))
        cut = max(1, int(len(order) * self.validation_fraction))
        # Validation is drawn from TRAIN rows only - it decides when to stop,
        # never what the test rows look like.
        valid, train = order[:cut], order[cut:]

        params = self._build(tokens.shape[2])
        moment = {name: np.zeros_like(value) for name, value in params.items()}
        velocity = {name: np.zeros_like(value) for name, value in params.items()}
        best, best_params, waited, step = np.inf, params, 0, 0

        for epoch in range(self.epochs):
            shuffled = rng.permutation(train)
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                _, grads = self.loss_and_grads(tokens[batch], target[batch], params)
                step += 1
                for name in params:
                    moment[name] = 0.9 * moment[name] + 0.1 * grads[name]
                    velocity[name] = 0.999 * velocity[name] + 0.001 * grads[name] ** 2
                    m_hat = moment[name] / (1 - 0.9 ** step)
                    v_hat = velocity[name] / (1 - 0.999 ** step)
                    params[name] = params[name] - self.learning_rate * m_hat / (
                        np.sqrt(v_hat) + 1e-8)
            predicted, _ = self.forward(tokens[valid], params)
            score = float(np.mean((predicted - target[valid]) ** 2))
            self.history.append(score)
            if score < best - 1e-6:
                best, best_params, waited = score, {k: v.copy() for k, v in params.items()}, 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.params = best_params

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.params:
            raise RuntimeError("fit before predict")
        tokens = self._tokens(np.asarray(X, dtype=np.float64))
        out, _ = self.forward(tokens, self.params)
        return out * self.y_std + self.y_mean

    def attention_map(self, X: np.ndarray) -> np.ndarray:
        """Mean attention weights, (scale reading) x (scale read).

        What the model decided the scales had to say to each other - the claim
        the architecture is there to make, so it is worth being able to look at.
        """
        tokens = self._tokens(np.asarray(X, dtype=np.float64))
        _, cache = self.forward(tokens, self.params)
        return cache["A"].mean(axis=0)
