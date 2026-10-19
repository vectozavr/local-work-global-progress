"""Pool seeds 0-9 and 10-19 for the one-route single agent, its best of three and the team, on map 1 or map 2; no model calls.
Usage: pool.py [map1|map2]  (map1 writes ./pooled, map2 writes ./pooled-map2)

Every run has already been replayed independently in its own campaign (summary.json for the single agent,
analysis-v2/summary.json for the team). This script checks that the seeds are distinct and complete (0-19),
copies the run reports, and recomputes the mean curves over all twenty runs with the same step rule.
Writes ./pooled/summary-single1.json, ./pooled/summary-team.json and ./pooled/curves-checked_seconds.json.
"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robot_routing.files import stats
from robot_routing.summary import _step_curve as step_curve

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2]/'data'/'routing-runs'
import sys
MAP = sys.argv[1] if len(sys.argv) > 1 else 'map1'
if MAP == 'map1':
    SINGLE = [RES/'multiseed-ninecars-single-v2-one-update-20261016', RES/'multiseed-ninecars-single-v2-one-update-seeds10-19-20261016']
    TEAM = [RES/'multiseed-ninecars-team-v2-20261016-p5', RES/'multiseed-ninecars-team-v2-seeds10-19-20261016-p5']
    BEST3 = [RES/'multiseed-ninecars-single-v2-best3-20261016', RES/'multiseed-ninecars-single-v2-best3-seeds10-19-20261016']
    OUT, INITIAL = 'pooled', 1476.
else:
    SINGLE = [RES/'map2-single-v2-one-update-20261016', RES/'map2-single-v2-one-update-seeds10-19-20261016']
    TEAM = [RES/'map2-team-v2-20261016-p5']
    BEST3 = [RES/'map2-single-v2-best3-20261016']
    OUT, INITIAL = 'pooled-map2', 1479.
read = lambda p: json.loads(Path(p).read_text())


def pool(paths, name):
    runs = []
    for p in paths:
        s = read(p); assert (s['all_planned_completed'] or '--partial' in sys.argv) and s['independent_replay_passed'], p
        runs += s['runs']
    seeds = sorted(r['trial']['replicate'] for r in runs); assert seeds == list(range(20)) or ('--partial' in sys.argv and seeds == list(range(len(seeds)))), (name, seeds)
    return sorted(runs, key=lambda r: r['trial']['replicate'])


def main():
    out = HERE/OUT; out.mkdir(exist_ok=True)
    single = pool([r/'summary.json' for r in SINGLE], 'single')
    team = pool([r/'analysis-v2/summary.json' for r in TEAM], 'team')
    best3 = pool([r/'summary.json' for r in BEST3], 'best3')
    safe = lambda r: r['final']['delivered'] == 9 and r['final']['collisions'] == 0
    for name, runs, extra in [('single1', single, ()), ('team', team, ('sequential_seconds', 'rounds')), ('best3', best3, ('rounds',))]:
        fields = {'final_F': [r['final']['F'] for r in runs], 'checked_seconds': [r['checked_seconds'] for r in runs],
                  'answers': [r['answers'] for r in runs], 'reported_cost_usd': [r['reported_cost_usd'] for r in runs]}
        for k in extra: fields[k] = [r[k] for r in runs]
        summary = dict(planned=20, completed=len(runs), all_planned_completed=len(runs) == 20, independent_replay_passed=True,
                       safe_complete=sum(map(safe, runs)), strict_target=sum(bool(r['reached_target']) for r in runs),
                       statistics={k: stats(v) for k, v in fields.items()}, reported_total_cost_usd=sum(fields['reported_cost_usd']),
                       sources=[str(p.relative_to(RES)) for p in dict(single1=SINGLE, team=TEAM, best3=BEST3)[name]], runs=runs)
        (out/f'summary-{name}.json').write_text(json.dumps(summary, indent=1) + '\n')
    curves = dict(single=step_curve(single, 'checked_seconds', INITIAL), team=step_curve(team, 'checked_seconds', INITIAL),
                  sequential=step_curve(team, 'sequential_seconds', INITIAL), best3=step_curve(best3, 'checked_seconds', INITIAL))
    (out/'curves-checked_seconds.json').write_text(json.dumps(curves) + '\n')
    for name in ('single1', 'team', 'best3'):
        s = read(out/f'summary-{name}.json'); st = s['statistics']
        print(name, 'safe', s['safe_complete'], 'target', s['strict_target'], 'F %.2f+-%.2f median %.1f' % (st['final_F']['mean'], st['final_F']['sample_sd'], st['final_F']['median']),
              'time %.1f+-%.1f median %.1f' % (st['checked_seconds']['mean'], st['checked_seconds']['sample_sd'], st['checked_seconds']['median']),
              'answers %.1f' % st['answers']['mean'], 'cost/run %.3f' % st['reported_cost_usd']['mean'],
              ('sequential %.1f+-%.1f' % (st['sequential_seconds']['mean'], st['sequential_seconds']['sample_sd'])) if 'sequential_seconds' in st else '')


if __name__ == '__main__':
    main()
