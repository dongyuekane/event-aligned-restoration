"""Run the finite-model comparison on archived synthetic cases."""
import argparse
import csv
import json
import random
import statistics
import time
from functools import lru_cache
from pathlib import Path

from frozen_policy import choose_action
from model import Config, Model, clear, state_view
from observations import NoisyModel


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
CASES = ARCHIVE / "cases.json"


class Search:

    def __init__(self, c, b, noise, method):
        self.m = NoisyModel(c, noise)
        self.fo = Model(c)
        self.method = method
        self.root = b
        self.expanded = 0
        self.pruned = 0

    @lru_cache(None)
    def lower(self, b, t, ld):
        s = sum((w[3] for w in b))
        return sum((w[3] / s * self.fo.optimal(((*w[:3], 1),), t, ld)[0] for w in b))

    @lru_cache(None)
    def optimal(self, b, t, ld):
        self.expanded += 1
        m = self.m
        best = (m.stop_cost(b, t, ld), 'stop')
        acts = []
        for a in m.actions(b, t, ld):
            im, oo = m.outcomes(b, t, ld, a)
            lower = im + sum((p * self.lower(bb, e, ll) for p, bb, e, ll, o in oo))
            acts.append((lower, a, im, oo))
        for lower, a, im, oo in sorted(acts, key=lambda v: v[0]):
            if lower >= best[0] - 1e-12:
                self.pruned += 1
                continue
            value = im + sum((p * self.optimal(bb, e, ll)[0] for p, bb, e, ll, o in oo))
            if value < best[0] - 1e-12:
                best = (value, a)
        return best

    @lru_cache(None)
    def root_prune(self, b, t, ld):
        """Candidate 1: same admissible lower bound, exact unpruned children."""
        m = self.m
        best = (m.stop_cost(b, t, ld), 'stop')
        ss = []
        for a in m.actions(b, t, ld):
            im, oo = m.outcomes(b, t, ld, a)
            ss.append((im + sum((p * self.lower(bb, e, ll) for p, bb, e, ll, o in oo)), a, im, oo))
        for lb, a, im, oo in sorted(ss, key=lambda z: z[0]):
            if lb >= best[0] - 1e-12:
                self.pruned += 1
                continue
            v = im + sum((p * m.optimal(bb, e, ll)[0] for p, bb, e, ll, o in oo))
            if v < best[0] - 1e-12:
                best = (v, a)
        return best

    @lru_cache(None)
    def action(self, b, t, ld):
        m = self.m
        if self.method == 'full_dp':
            return m.optimal(b, t, ld)[1]
        if self.method == 'bound_recursive':
            return self.optimal(b, t, ld)[1]
        if self.method == 'bound_root':
            return self.root_prune(b, t, ld)[1]
        if self.method == 'two_step':
            return m.policy_action('two_step_voi_rollout', b, t, ld)
        if self.method == 'three_step':
            vals = [(m.stop_cost(b, t, ld), 'stop')]
            for a in m.actions(b, t, ld):
                im, oo = m.outcomes(b, t, ld, a)
                vals.append((im + sum((p * m.rollout_value(bb, e, ll, 2) for p, bb, e, ll, o in oo)), a))
            return min(vals, key=lambda z: z[0])[1]
        if self.method == 'frozen_analytic':
            return choose_action(state_view(m, b, t, ld))
        raise ValueError(self.method)

    @lru_cache(None)
    def value(self, b, t, ld):
        m = self.m
        a = self.action(b, t, ld)
        if a == 'stop':
            return m.stop_cost(b, t, ld)
        im, oo = m.outcomes(b, t, ld, a)
        return im + sum((p * self.value(bb, e, ll) for p, bb, e, ll, o in oo))

    def close(self):
        for n in ('lower', 'optimal', 'root_prune', 'action', 'value'):
            getattr(self, n).cache_clear()
        clear(self.m)
        self.m.outcomes.cache_clear()
        clear(self.fo)
METHODS = ['frozen_analytic', 'two_step', 'three_step', 'full_dp', 'bound_root', 'bound_recursive']

def run(split, output, smoke=False):
    cases = [case for case in json.loads(CASES.read_text()) if case['split'] == split]
    if smoke:
        cases = cases[:2]
    rows = []
    start = time.perf_counter()
    for x in cases:
        c = Config(**x['config'])
        b = tuple(map(tuple, x['belief']))
        for method in METHODS:
            s = Search(c, b, x['noise'], method)
            tic = time.perf_counter()
            v = s.value(b, 0, False) * 1000
            sec = time.perf_counter() - tic
            rows.append({'case': c.case_id, 'split': split, 'method': method, 'noise': x['noise'] / 10, 'ens_kwh': v, 'policy_evaluation_s': sec, 'expanded_pruned_nodes': s.expanded, 'pruned_actions': s.pruned, 'root_action': s.action(b, 0, False)})
            s.close()
        ref = next((r['ens_kwh'] for r in rows[-6:] if r['method'] == 'full_dp'))
        assert all((r['ens_kwh'] >= ref - 1e-07 for r in rows[-6:]))
        assert all((abs(r['ens_kwh'] - ref) < 1e-07 for r in rows[-6:] if r['method'].startswith('bound')))
        print(c.case_id, round(ref, 6), round(time.perf_counter() - start, 2), flush=True)
    with (output / (split + '_results.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    (output / (split + '_metrics.json')).write_text(json.dumps(summarize(rows), indent=2))

def summarize(rows):
    refs = {r['case']: r['ens_kwh'] for r in rows if r['method'] == 'full_dp'}
    out = {}
    for meth in METHODS:
        rr = [r for r in rows if r['method'] == meth]
        g = [r['ens_kwh'] - refs[r['case']] for r in rr]
        out[meth] = {'n': len(rr), 'mean_kwh': statistics.mean((r['ens_kwh'] for r in rr)), 'mean_regret_kwh': statistics.mean(g), 'max_regret_kwh': max(g), 'exact_matches': sum((abs(v) < 1e-07 for v in g)), 'total_evaluation_s': sum((r['policy_evaluation_s'] for r in rr))}
    return out

def timing(output):
    cases = [x for x in json.loads(CASES.read_text()) if x['split'] == 'confirmation']
    rows = []
    rng = random.Random(2026111455)
    for rep in range(3):
        for x in cases:
            methods = METHODS.copy()
            rng.shuffle(methods)
            c = Config(**x['config'])
            b = tuple(map(tuple, x['belief']))
            for meth in methods:
                tic = time.perf_counter_ns()
                s = Search(c, b, x['noise'], meth)
                a = s.action(b, 0, False)
                ns = time.perf_counter_ns() - tic
                rows.append({'case': c.case_id, 'rep': rep, 'method': meth, 'elapsed_ns': ns, 'action': a})
                s.close()
    with (output / 'timing.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    out = {m: {'n': len([r for r in rows if r['method'] == m]), 'median_ms': statistics.median((r['elapsed_ns'] / 1000000.0 for r in rows if r['method'] == m)), 'mean_ms': statistics.mean((r['elapsed_ns'] / 1000000.0 for r in rows if r['method'] == m))} for m in METHODS}
    (output / 'timing_metrics.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "development", "confirmation", "timing"])
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    output = options.output.resolve()
    if output == ROOT or output.is_relative_to(ARCHIVE):
        parser.error("Choose an output directory outside the archived measurements.")
    output.mkdir(parents=True, exist_ok=True)
    if options.mode == "timing":
        timing(output)
    else:
        run("confirmation" if options.mode == "smoke" else options.mode, output, options.mode == "smoke")
