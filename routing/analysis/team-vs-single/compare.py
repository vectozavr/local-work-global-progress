"""Permutation tests and curve crossings between the pooled protocols of one map; no model calls.
Usage: compare.py [map1|map2]"""
import json, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
MAP = sys.argv[1] if len(sys.argv) > 1 else 'map1'
P = HERE/('pooled' if MAP == 'map1' else 'pooled-map2'); FS = 144. if MAP == 'map1' else 147.
read = lambda p: json.loads(Path(p).read_text())
runs = {k: read(P/f'summary-{k}.json')['runs'] for k in ('single1', 'best3', 'team')}
T = {k: np.array([r['checked_seconds'] for r in v]) for k, v in runs.items()}
rng = np.random.default_rng(2)
def perm(a, b, stat, B=100000):
    allv = np.concatenate([a, b]); n = len(a); obs = stat(a, b); k = 0
    for _ in range(B):
        p = rng.permutation(allv); k += abs(stat(p[:n], p[n:])) >= abs(obs) - 1e-12
    return float(obs), k / B
out = {}
for k in ('single1', 'best3'):
    a, b = T[k], T['team']
    out[k] = dict(mean_diff=perm(a, b, lambda u, v: u.mean() - v.mean()), median_diff=perm(a, b, lambda u, v: np.median(u) - np.median(v)),
                  sd_ratio=perm(a, b, lambda u, v: np.log(u.std(ddof=1) / v.std(ddof=1))), team_faster=int((b < a).sum()), over700=int((a > 700).sum()))
    out[k]['sd_ratio'] = (float(np.exp(out[k]['sd_ratio'][0])), out[k]['sd_ratio'][1])
out['team_over700'] = int((T['team'] > 700).sum()); out['team_max'] = float(T['team'].max())
c = read(P/'curves-checked_seconds.json')
def at(cv, t):
    i = np.searchsorted(np.asarray(cv['seconds']), t, side='right') - 1; return cv['mean'][i]
def changes(a, b):
    xs = sorted(set(c[a]['seconds']) | set(c[b]['seconds'])); prev = None; ch = []
    for t in xs:
        s = at(c[a], t) < at(c[b], t)
        if prev is not None and s != prev: ch.append(round(t))
        prev = s
    return ch
out['team_below_single_changes'] = changes('team', 'single'); out['team_below_best3_changes'] = changes('team', 'best3')
def first_below(cv, g):
    for x, m in zip(cv['seconds'], cv['mean']):
        if m - FS <= g: return round(x)
out['first_mean_gap_below'] = {g: {k: first_below(c[k], g) for k in ('team', 'best3', 'single')} for g in (50, 20, 10)}
out['final_mean_gap'] = {k: round(c[k]['mean'][-1] - FS, 2) for k in ('team', 'best3', 'single', 'sequential')}
(HERE/f'compare-{MAP}.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out))
