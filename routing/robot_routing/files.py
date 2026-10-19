"""Small helpers for run records: JSON files, JSONL journals, checksums and summary statistics."""
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()] if path and Path(path).exists() else []


def save(path, data):
    path = Path(path); temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n'); temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values),
                sample_sd=statistics.stdev(values) if len(values) > 1 else None) if values else None


def step_curves(reports, clock='checked_seconds', initial=986., timeline=None):
    """Mean, minimum and maximum loss over runs against time; a run holds its plan until the next accepted change."""
    import numpy as np
    x = np.array(sorted({0.} | set(timeline or []) | {r[clock] for report in reports for r in report['trace']} |
                        {report[clock] for report in reports}), dtype=float)
    curves = []
    for report in reports:
        t = np.array([0.] + [r[clock] for r in report['trace']])
        y = np.array([initial] + [r['F'] for r in report['trace']])
        curves.append(y[np.searchsorted(t, x, side='right') - 1])
    matrix = np.array(curves)
    return dict(seconds=x.tolist(), mean=matrix.mean(axis=0).tolist(), minimum=matrix.min(axis=0).tolist(),
                maximum=matrix.max(axis=0).tolist(), per_run=matrix.tolist(), runs=len(reports),
                terminal_extension='Hold final installed loss; no further API work')
