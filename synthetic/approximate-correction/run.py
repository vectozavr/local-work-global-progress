"""Approximate-correction study: does a persistent correction error leave a floor?

Same fixed six-block, eight-bit task, randomized coordinate solver, singleton
verification and best-color merge as synthetic/known-correction/run.py. The task
(targets, weights, private targets, activity rates) is read from that study's saved
results, so it is identical. No LLM calls; NumPy only; all costs are modeled.

The only change: agents search a corrected objective whose coefficients are the exact
conditional coefficients plus a fixed bias eps_bj = eta * s_bj, with independent random
signs s_bj in {-1, +1} drawn once per run and unknown to the agents. The corrected
objective stays anchored (affine in u - x_b). For every search we record the two-point
correction errors of Theorem 2 (at a true block minimizer and at the returned proposal),
the realized fraction Q of the corrected improvement, and the exact block improvements,
and propagate the per-run recurrence used for the routing study (Appendix E.8).
"""
from pathlib import Path
import json, math, hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
TASK_FILE = HERE.parent / 'known-correction' / 'results.json'
task = json.loads(TASK_FILE.read_text())
target = np.array(task['target']); weights = np.array(task['weights'])
private = np.array(task['private_target']); rates = np.array(task['rates'])
N, D, W = task['n'], task['d'], task['coupling']
FSTAR, MU = task['optimum'], task['mu']
REPS, EPOCHS, SEED = 2000, 160, 20261009
ETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
HS = [8, 32]
COLORS = [np.arange(0, N, 2), np.arange(1, N, 2)]
KAPPA = len(COLORS)


def objective(x):
    return ((x != target) * weights).sum(axis=(-2, -1)) + W * (x[..., :-1, :] != x[..., 1:, :]).sum(axis=(-2, -1))


def coefficients(x):
    """Exact coefficient of u_j in the global conditional change of block b."""
    a = np.broadcast_to(weights * (1 - 2 * target), x.shape).copy()
    a[..., :-1, :] += W * (1 - 2 * x[..., 1:, :])
    a[..., 1:, :] += W * (1 - 2 * x[..., :-1, :])
    return a


# Sanity: the affine coefficients reproduce exact singleton changes.
_rng = np.random.default_rng(7)
for _ in range(200):
    x = _rng.integers(0, 2, (N, D)); b = int(_rng.integers(N)); u = _rng.integers(0, 2, D)
    y = x.copy(); y[b] = u
    assert abs(float(np.dot(coefficients(x)[b], u - x[b])) - float(objective(y) - objective(x))) < 1e-10
assert abs(float(objective(target)) - FSTAR) < 1e-10
min_abs_coefficient = float(min(np.abs(coefficients(np.array(bits).reshape(N, D))).min()
                                for bits in _rng.integers(0, 2, (2000, N * D))))


def trial(H, eta):
    rng = np.random.default_rng(SEED + 1000 * H + int(round(100 * eta)))
    x = np.broadcast_to(1 - target, (REPS, N, D)).copy()
    eps = eta * rng.choice([-1.0, 1.0], size=(REPS, N, D))
    E = (objective(x) - FSTAR).astype(float)          # per-run recurrence envelope
    gaps, bounds = [float(np.mean(E))], [float(np.mean(E))]
    max_err_star = np.zeros(N); max_err_U = np.zeros(N)
    mean_err_star, mean_err_U = [], []
    r, b_idx = np.indices((REPS, N))
    for epoch in range(EPOCHS):
        before = objective(x); base = x.copy()
        A = coefficients(x); Ahat = A + eps
        # Private search on the biased corrected objective (anchored, affine).
        y = x.copy()
        for _ in range(H):
            j = rng.integers(D, size=(REPS, N)); success = rng.random((REPS, N)) < rates
            choice = (Ahat[r, b_idx, j] < 0).astype(int)
            y[r, b_idx, j] = np.where(success, choice, y[r, b_idx, j])
        # Verification and best-color merge (unchanged protocol).
        g = np.zeros((REPS, N))
        for b in range(N):
            cand = x.copy(); cand[:, b] = y[:, b]
            g[:, b] = np.maximum(0, before - objective(cand))
        sums = np.stack([g[:, c].sum(1) for c in COLORS], axis=1); color = np.argmax(sums, axis=1)
        for ci, c in enumerate(COLORS):
            for b in c:
                use = (color == ci) & (g[:, b] > 1e-10)
                x[use, b] = y[use, b]
        after = objective(x)
        assert np.max(after - before) < 1e-9
        # Two-point errors, realized progress and exact block improvements at the base.
        ustar = (A < 0).astype(int)                    # true block minimizer per block
        Dexact = -((A * (ustar - base)).sum(-1))        # best true block decrease >= 0
        err_star = np.abs((eps * (ustar - base)).sum(-1))
        err_U = np.abs((eps * (y - base)).sum(-1))
        fhat_U = (Ahat * (y - base)).sum(-1)
        fhat_star = (Ahat * ((Ahat < 0).astype(int) - base)).sum(-1)   # minimum of the corrected objective
        Q = np.where(fhat_star < -1e-12, fhat_U / np.where(fhat_star < -1e-12, fhat_star, 1), 0.0)
        assert np.all(Q > -1e-9) and np.all(Q < 1 + 1e-9)
        lower = np.maximum(0, Q * (Dexact - err_star) - err_U)
        assert np.all(g + 1e-9 >= lower)               # pathwise local-gain inequality
        committed = np.max(np.stack([lower[:, c].sum(1) for c in COLORS], 1), 1)
        assert np.all(before - after + 1e-9 >= committed)
        E = np.minimum(E, np.maximum(0, E - committed))
        assert np.all(after - FSTAR <= E + 1e-9)
        max_err_star = np.maximum(max_err_star, err_star.max(0)); max_err_U = np.maximum(max_err_U, err_U.max(0))
        mean_err_star.append(float(err_star.mean())); mean_err_U.append(float(err_U.mean()))
        gaps.append(float(np.mean(after - FSTAR))); bounds.append(float(np.mean(E)))
    final = objective(x) - FSTAR
    tail = gaps[-20:]
    return dict(H=H, eta=eta, repetitions=REPS, epochs=EPOCHS,
                mean_gap_by_epoch=gaps, mean_recurrence_bound_by_epoch=bounds,
                final_mean_gap=float(final.mean()), final_gap_se=float(final.std(ddof=1) / math.sqrt(REPS)),
                solved=int((final <= 1e-9).sum()), plateau_drift_last20=float(tail[0] - tail[-1]),
                final_mean_recurrence_bound=bounds[-1],
                mean_err_star_by_epoch=mean_err_star, mean_err_U_by_epoch=mean_err_U,
                max_err_star=float(max_err_star.max()), max_err_U=float(max_err_U.max()))


rows = [trial(H, eta) for H in HS for eta in ETAS]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = dict(seed=SEED, repetitions=REPS, epochs=EPOCHS, etas=ETAS, horizons=HS, kappa=KAPPA,
              optimum=FSTAR, mu=MU, min_abs_exact_coefficient_sampled=min_abs_coefficient,
              bias_model='eps_bj = eta * s_bj, s_bj in {-1,+1} uniform, fixed per run, unknown to agents',
              recurrence='E_{k+1} = min(E_k, max(0, E_k - max_color sum_b max(0, Q_b (D_b - e*_b) - e^U_b)))',
              task_source=str(TASK_FILE.relative_to(HERE.parent.parent)), task_sha256=sha(TASK_FILE),
              model_calls=0, rows=rows,
              scope='Same synthetic task and protocol as known-correction; only the correction is biased. '
                    'No LLM calls; modeled quantities only.')
(HERE / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
for row in rows:
    print(f"H={row['H']:3d} eta={row['eta']:.2f} final gap={row['final_mean_gap']:7.3f}+-{row['final_gap_se']:.3f} "
          f"solved={row['solved']:4d} bound={row['final_mean_recurrence_bound']:7.3f} drift={row['plateau_drift_last20']:.4f} "
          f"maxe*={row['max_err_star']:.2f} maxeU={row['max_err_U']:.2f}")
