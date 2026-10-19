# Execution feedback in SQLite

Section 8.2 and Appendix F of the paper. A join-order controller for SQLite chooses, for every query of a stream,
from which side to start a two-hop join. Its correction is a table of edge counts that it learns from the engine's
own execution counters (`NVISIT` scan status), instead of measuring every alternative in advance. The study asks
when acquiring correction information this way pays off. The controller is a deterministic routine, not a language
model, and nothing here calls an API.

## Requirements

Python 3.9 or newer (standard library only), a C compiler, and about 1 GB of free disk for the generated databases.
`build_sqlite.sh` downloads the official SQLite 3.46.1 amalgamation (public domain), checks its SHA-256 (recorded in
`vendor/provenance.json`) and builds it twice: with statement scan status, and additionally with STAT4 statistics.
The build works on macOS and Linux.

## Run

From this folder:

```bash
sh build_sqlite.sh              # download and build vendor/libsqlite_profile and vendor/libsqlite_stat4
python confirm.py               # build three databases, check every answer against an independent reference
python stat4_control.py         # control: the native planner with STAT4 statistics
python profile_cost_control.py  # the timed comparison of Table F.2 and Figure F.1
python audit_feedback.py        # replays the 483 feedback measurements
```

The results go to `results/`: `instrumentation-results.json` and `instrumentation-summary.json` (324 timed runs of
twelve policies on three workloads: Table F.2 and Figures 5(a) and F.1 of the paper), `feedback-audit.json` (replay of
every feedback measurement), `confirmation-results.json` (database construction and correctness checks) and
`stat4-results.json` (the STAT4 control). VM steps and decisions are deterministic; absolute times depend on the
machine.

`join_study.py` generates the data and implements the policies; `sqlite_profile.py` wraps the compiled library.
The databases are generated and analyzed with Python's built-in `sqlite3`; only the timed queries run on the
pinned build.
