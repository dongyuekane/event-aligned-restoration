"""Observation likelihoods for repeated imperfect diagnoses."""
from functools import lru_cache

from model import Model, canonical


class NoisyModel(Model):

    def __init__(self, c, err):
        super().__init__(c)
        self.err = err

    @lru_cache(None)
    def outcomes(self, b, t, ld, a):
        if a != 'diagnose' or not self.err:
            return super().outcomes(b, t, ld, a)
        groups = {}
        total = sum((w[3] for w in b))
        im = 0.0
        for mode, due, ack, weight in b:
            end, ll, ww, obs, cost = self.transition((mode, due, ack), t, ld, a)
            im += weight * cost / total
            branches = [(obs, 10)]
            if obs[1] in ('temporary_block', 'persistent_block'):
                other = 'persistent_block' if obs[1] == 'temporary_block' else 'temporary_block'
                branches = [(obs, 10 - self.err), ((obs[0], other, obs[2]), self.err)]
            for oo, w in branches:
                groups.setdefault((end, ll, oo), []).append((*ww, weight * w))
        return (im, tuple(((sum((r[3] for r in rows)) / (10 * total), canonical(rows), end, ll, oo) for (end, ll, oo), rows in sorted(groups.items()))))
