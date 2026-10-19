"""Per-campaign numbers for maps 1 and 2 (safe runs, final loss, search time, answers, cost, curve crossings), from
the replayed summaries; no model calls. Writes numbers.json."""
import json, statistics, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2]/'data'/'routing-runs'
read = lambda p: json.loads(Path(p).read_text())
CAMPAIGNS = dict(single1=HERE/'pooled/summary-single1.json',
                 single3=RES/'multiseed-ninecars-single-v2-exact-three-20261016/summary.json',
                 team=HERE/'pooled/summary-team.json',
                 best3=HERE/'pooled/summary-best3.json',
                 nocorr=RES/'multiseed-ninecars-team-v2-no-correction-20261016-p5/analysis-v2/summary.json')


def ms(v, d=2):
    return f'${statistics.mean(v):.{d}f}\\pm{statistics.stdev(v):.{d}f}$'


def runs(name):
    s = read(CAMPAIGNS[name]); rs = sorted(s['runs'], key=lambda r: r['trial']['replicate'])
    assert len(rs) in (10, 20) and s['independent_replay_passed'], name
    return s, rs


def safe(r):
    return r['final']['delivered'] == 9 and r['final']['collisions'] == 0


def first_target_time(r):
    for row in r['trace']:
        if row['F'] <= 153 and row.get('delivered', 9) == 9 and row.get('collisions', 0) == 0:
            return row['checked_seconds']
    return None


def crossing(curve_a, curve_b):
    """Earliest time after which curve_a's mean stays at or below curve_b's mean."""
    xs = sorted(set(curve_a['seconds']) | set(curve_b['seconds']))
    def at(c, t):
        i = np.searchsorted(np.asarray(c['seconds']), t, side='right') - 1; return c['mean'][i]
    last_above = None
    for t in xs:
        if at(curve_a, t) > at(curve_b, t): last_above = t
    later = [t for t in xs if last_above is None or t > last_above]
    return later[0] if later else None


def main():
    out = {}; S = {k: runs(k) for k in CAMPAIGNS}
    for k, (s, rs) in S.items():
        F = [r['final']['F'] for r in rs]; T = [r['checked_seconds'] for r in rs]
        ans = [r.get('answers', r.get('checked_attempts')) for r in rs]
        cost = [r['reported_cost_usd'] for r in rs]
        out[k] = dict(safe=sum(map(safe, rs)), target=sum(bool(r['reached_target']) for r in rs),
                      F_mean=statistics.mean(F), F_sd=statistics.stdev(F), F_median=statistics.median(F),
                      time_mean=statistics.mean(T), time_sd=statistics.stdev(T), time_median=statistics.median(T),
                      answers_mean=statistics.mean(ans), answers_sd=statistics.stdev(ans),
                      cost_mean=statistics.mean(cost), cost_sd=statistics.stdev(cost), cost_total=sum(cost),
                      time_to_target=[first_target_time(r) for r in rs], per_seed_F=F, per_seed_time=T, per_seed_answers=ans)
        if k in ('team', 'nocorr', 'best3'):
            rounds = [r['rounds'] for r in rs]; seq = [r.get('sequential_seconds', r['checked_seconds']) for r in rs]
            out[k].update(rounds_mean=statistics.mean(rounds), rounds_sd=statistics.stdev(rounds), per_seed_rounds=rounds,
                          sequential_mean=statistics.mean(seq), sequential_sd=statistics.stdev(seq), per_seed_sequential=seq,
                          collisions=[r['final']['collisions'] for r in rs],
                          collision_free_part=[r['final']['F'] - 100 * r['final']['collisions'] for r in rs])
    # curves and crossings
    pooled = read(HERE/'pooled/curves-checked_seconds.json'); c_single1 = pooled['single']
    c_single3 = read(RES/'multiseed-ninecars-single-v2-exact-three-20261016/curves-checked_seconds.json')['single_v2_exact3']
    c_team = pooled['team']
    out['crossing_team_below_single1'] = crossing(c_team, c_single1)
    out['crossing_team_below_single3'] = crossing(c_team, c_single3)
    for k in ('single1', 'single3'):
        out[f'team_faster_than_{k}_seeds'] = [sum(a < b for a, b in zip(out['team']['per_seed_time'], out[k]['per_seed_time'])), min(len(out['team']['per_seed_time']), len(out[k]['per_seed_time']))]
    (HERE/'numbers.json').write_text(json.dumps(out, indent=1) + '\n')
    summary = {k: dict(safe=out[k]['safe'], target=out[k]['target'], F=f"{out[k]['F_mean']:.2f}+-{out[k]['F_sd']:.2f}",
                       time=f"{out[k]['time_mean']:.1f}+-{out[k]['time_sd']:.1f}", cost=round(out[k]['cost_mean'], 3)) for k in CAMPAIGNS}
    print(json.dumps(dict(summary=summary, crossing_single1=out['crossing_team_below_single1'], crossing_single3=out['crossing_team_below_single3'],
                          team_faster_seeds=(out['team_faster_than_single1_seeds'], out['team_faster_than_single3_seeds']),
                          sequential=f"{out['team']['sequential_mean']:.1f}+-{out['team']['sequential_sd']:.1f}",
                          nocorr_collision_free=statistics.mean(out['nocorr']['collision_free_part']))))


if __name__ == '__main__':
    main()
