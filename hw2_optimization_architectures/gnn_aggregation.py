"""HW2 -- message-passing GNN aggregation constructions (universal aggregation).

The written questions ask you to construct the two functions f, g so that the
generic aggregator

    generic_{f,g}({h_u : u in N(v)}) = f( sum_{u in N(v)} g(h_u) )

realises (a) mean aggregation and (b) max aggregation. This module implements
those constructions and verifies them numerically on random neighbourhoods, and
implements a small message-passing GNN layer using each aggregator.
"""
from __future__ import annotations

import numpy as np


# ---- (a) mean aggregation -------------------------------------------------
# Trick: append a constant 1 to each message so the sum also counts the degree,
# then divide the summed features by that count inside f.
def g_mean(h: np.ndarray) -> np.ndarray:
    return np.concatenate([h, np.ones(1)])


def f_mean(s: np.ndarray) -> np.ndarray:
    feat, count = s[:-1], s[-1]
    return feat / (count + 1e-12)


def aggregate_mean(neighbours: np.ndarray) -> np.ndarray:
    s = sum(g_mean(h) for h in neighbours)
    return f_mean(s)


# ---- (b) max aggregation (approximate, via a soft-max / L_inf trick) -------
# For a large exponent p,  (sum_u exp(p * h_u))^(1/p) -> max_u h_u  coordinatewise
# (a smooth approximation of the L_inf norm). We keep values finite by working in
# a numerically stable way. g maps h -> exp(p*h), f maps s -> log(s)/p.
class MaxAggregator:
    def __init__(self, p: float = 40.0):
        self.p = p

    def g(self, h: np.ndarray) -> np.ndarray:
        return np.exp(self.p * h)

    def f(self, s: np.ndarray) -> np.ndarray:
        return np.log(s + 1e-30) / self.p

    def __call__(self, neighbours: np.ndarray) -> np.ndarray:
        # stable log-sum-exp: subtract the per-coordinate max
        H = np.stack(neighbours, axis=0)              # (deg, m)
        m = H.max(axis=0)
        s = np.exp(self.p * (H - m)).sum(axis=0)
        return m + np.log(s + 1e-30) / self.p


# ---- a small message-passing GNN layer using these aggregators ------------
class MPLayer:
    """One round of message passing: update h_v from aggregated neighbours."""

    def __init__(self, aggregator="mean", p=40.0, seed=0):
        self.aggregator = aggregator
        self.max_agg = MaxAggregator(p)

    def aggregate(self, neighbours):
        if len(neighbours) == 0:
            return None
        if self.aggregator == "mean":
            return aggregate_mean(neighbours)
        return self.max_agg(neighbours)

    def forward(self, adj: np.ndarray, feats: np.ndarray) -> np.ndarray:
        n = feats.shape[0]
        out = feats.copy()
        for v in range(n):
            nbrs = [feats[u] for u in range(n) if adj[v, u] and u != v]
            agg = self.aggregate(nbrs)
            if agg is not None:
                out[v] = agg
        return out


def _verify():
    rng = np.random.default_rng(0)
    ok_mean = True
    ok_max = True
    max_agg = MaxAggregator(p=50.0)
    for _ in range(500):
        deg = rng.integers(2, 8)
        nbrs = [rng.standard_normal(4) for _ in range(deg)]
        # mean
        got = aggregate_mean(nbrs)
        exp = np.mean(nbrs, axis=0)
        ok_mean &= np.allclose(got, exp, atol=1e-6)
        # max
        gotm = max_agg(nbrs)
        expm = np.max(nbrs, axis=0)
        ok_max &= np.allclose(gotm, expm, atol=5e-2)
    print(f"mean aggregation exact: {ok_mean}")
    print(f"max aggregation (soft, p=50) within 5e-2 of true max: {ok_max}")

    # message-passing sanity: mean-agg on a small graph
    adj = np.array([[0, 1, 1, 0],
                    [1, 0, 1, 0],
                    [1, 1, 0, 1],
                    [0, 0, 1, 0]])
    feats = rng.standard_normal((4, 3))
    layer = MPLayer("mean")
    out = layer.forward(adj, feats)
    # node 3 has a single neighbour (node 2) -> its update equals feats[2]
    assert np.allclose(out[3], feats[2], atol=1e-6)
    print("message-passing layer: single-neighbour update matches neighbour  OK")
    assert ok_mean and ok_max
    print("All HW2 GNN aggregation checks passed.")


if __name__ == "__main__":
    _verify()
