"""Recompute the robot-routing results of the paper from the recorded runs; no model calls.

Reads the run records in data/routing-runs/ and the audited constants in data/routing-theory/, and writes the numbers
reported in the paper as JSON files next to the scripts (listed at the end). Takes about two minutes.

    python routing/analysis/run_all.py
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = '../../../data/routing-runs'
THEORY = '../../../data/routing-theory'

STEPS = [
    # maps 1 and 2: pooled summaries, permutation tests, further numbers of the text
    ('team-vs-single', ['pool.py', 'map1']), ('team-vs-single', ['pool.py', 'map2']),
    ('team-vs-single', ['compare.py', 'map1']), ('team-vs-single', ['compare.py', 'map2']),
    ('team-vs-single', ['final_loss_tests.py']),
    ('team-vs-single', ['extras.py', '--team', f'{RES}/multiseed-ninecars-team-v2-20261016-p5',
                        f'{RES}/multiseed-ninecars-team-v2-seeds10-19-20261016-p5', '--theory', f'{THEORY}/map1-team', '--out', 'extras-map1.json']),
    ('team-vs-single', ['extras.py', '--team', f'{RES}/map2-team-v2-20261016-p5', '--theory', f'{THEORY}/map2-team', '--out', 'extras-map2.json']),
    ('team-vs-single', ['summary_numbers.py']),
    # two floors (composed from the runs of the two maps) and the merge rules
    ('two-floors', ['compose_two_floor_team.py']),
    ('merge-rules', ['compose_and_compare.py']),
    ('two-floors', ['stats_two_floor.py']),
    ('merge-rules', ['merge_numbers.py']),
    ('merge-rules', ['time_to_safe.py']),
    ('merge-rules', ['conflict_dynamics.py']),
    ('merge-rules', ['realized_conflict_bound.py']),
]

OUTPUTS = {
    'team-vs-single/pooled/, pooled-map2/': 'per-run summaries and mean curves of the single agent, best-of-3 and team',
    'team-vs-single/compare-map1.json, compare-map2.json': 'search-time tests and curve crossings',
    'team-vs-single/final-loss-tests.json': 'final-loss tests and threshold counts',
    'team-vs-single/numbers.json': 'per-campaign numbers of maps 1 and 2',
    'team-vs-single/extras-map1.json, extras-map2.json': 'coupling during a run, correction error by round, best-block bound',
    'two-floors/composed/, numbers-two-floor.json': 'two-floor team, its sequential replay and the tests',
    'merge-rules/comparison.json, merge-numbers.json': 'Algorithm 2 and the unchecked merge against Algorithm 1',
    'merge-rules/time-to-safe.json': 'time to the first safe plan',
    'merge-rules/conflict-dynamics.json': 'conflicts among improving proposals, by round and phase',
    'merge-rules/realized-conflict-bound.json': 'the round-by-round bound with kappa, min{kappa,|I^k|}, kappa_k and kappa*_k',
}


def main():
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    for folder, cmd in STEPS:
        print(f'[{folder}] {" ".join(cmd)}', flush=True)
        done = subprocess.run([sys.executable, *cmd], cwd=HERE/folder, env=env, stdout=subprocess.DEVNULL)
        if done.returncode:
            sys.exit(f'step failed: {folder}/{cmd[0]}')
    print('\nResults in routing/analysis/:')
    for path, what in OUTPUTS.items():
        print(f'  {path:52s} {what}')


if __name__ == '__main__':
    main()
