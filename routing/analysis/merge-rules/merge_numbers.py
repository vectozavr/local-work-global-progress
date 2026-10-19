"""Safe runs, threshold, final loss, search and sequential time, answers, cost and rounds of the merge-rule variants
(best combination, team unsafe) on every task, from the replayed summaries; no model calls. Writes merge-numbers.json."""
import json, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent; RES = HERE.parents[2]/'data'/'routing-runs'
read = lambda p: json.loads(Path(p).read_text())
CAMP = {'best_subset': 'team-best-combination', 'all': 'team-unsafe'}
ROBOTS = {1: 9, 2: 9, 3: 18}


def runs(mode, m):
    if m == 3: return read(HERE/f'composed/summary-{mode}.json')['runs']
    s = read(RES/f'{CAMP[mode]}-map{m}-20261017/analysis-v2/summary.json'); assert s['independent_replay_passed']
    return sorted(s['runs'], key=lambda r: r['trial']['replicate'])


def mean_sd(v): return [statistics.mean(v), statistics.stdev(v)]


nums = {}
for m, task in ((1, 'map1'), (2, 'map2'), (3, 'two-floors')):
    for mode in CAMP:
        r = runs(mode, m)
        nums[f'{task}/{mode}'] = dict(
            runs=len(r), safe=sum(x['final']['delivered'] == ROBOTS[m] and x['final']['collisions'] == 0 for x in r),
            threshold=sum(bool(x['reached_target']) for x in r), final_F=mean_sd([x['final']['F'] for x in r]),
            search_seconds=mean_sd([x['checked_seconds'] for x in r]), sequential_seconds=mean_sd([x['sequential_seconds'] for x in r]),
            answers=mean_sd([x['answers'] for x in r]), cost_usd=mean_sd([x['reported_cost_usd'] for x in r]),
            rounds=statistics.mean(x['rounds'] for x in r))
(HERE/'merge-numbers.json').write_text(json.dumps(nums, indent=1) + '\n')
for k, v in nums.items():
    print(k, {kk: ([round(x, 3) for x in vv] if isinstance(vv, list) else round(vv, 2)) for kk, vv in v.items()})
