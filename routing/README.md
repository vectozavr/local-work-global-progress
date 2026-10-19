# Robot routing with LLM agents

The main experiment of the paper (Section 8.1, Appendices C and E). Agents write explicit command lists for nine
robots on a 16×16 grid with walls, over 48 ticks; the loss adds the robots' arrival ticks and a penalty of 100 per
collision. A **team** of three agents, each controlling three robots, follows the protocol of the paper: in every
round the agents search their own robots in parallel against a frozen plan, each for up to three answers, with a
correction for the collisions with other agents' robots that the coordinator has already found; the coordinator then
verifies the proposals and accepts the best one (Algorithm 1) or the best checked combination of them (Algorithm 2).
Baselines are a **single agent** that controls all robots and changes one route per answer, and **best-of-W**:
W single agents asked in parallel from the same plan. The **two-floor** task joins both maps into one task with
eighteen robots.

All agents are GPT-OSS-120B (medium reasoning, temperature 0.6, top-p 0.95), served by Groq through OpenRouter.

## Layout

| Path | Content |
|---|---|
| `robot_routing/world.py` | the grid world: routes, collisions and the loss |
| `robot_routing/prompt_team.py`, `prompt_single.py` | the prompts of a team agent and of a single agent |
| `robot_routing/two_floors.py` | the two-floor task and its prompt |
| `robot_routing/team.py` | the team protocol: private search, correction, verification, merge |
| `robot_routing/single.py`, `single_two_floors.py` | the single agent and best-of-W |
| `robot_routing/llm.py` | the OpenRouter client; extraction of the command object; re-asking malformed answers |
| `robot_routing/reference.py` | exact block minima (conflict-based search) and single-robot best responses |
| `robot_routing/summary.py` | independent replay of finished runs and campaign summaries |
| `maps/` | the two maps and the two-floor task, each with a certified optimal plan (never shown to an agent) |
| `campaigns/` | one file per campaign of the paper: map, protocol, model, stopping rule, seeds |
| `run_campaign.py` | runs a campaign |
| `show_prompt.py` | prints the first prompt of a campaign without calling a model |
| `analysis/` | statistics and tests of the paper, and the audit of the theory's constants |

The records of the runs of the paper are in `../data/` (see the main README).

## Run a campaign

You need an [OpenRouter](https://openrouter.ai) API key in a `.env` file at the repository root (see `.env.example`)
or in the environment:

```bash
export OPENROUTER_API_KEY=...
```

```bash
python routing/run_campaign.py --list
python routing/show_prompt.py map1-team
python routing/run_campaign.py map1-team --seeds 0-2 --execute
```

Runs go to `runs/<campaign>/` (or `--out`): for every run, each prompt, answer and returned reasoning
(`protocol/attempts/`), the raw request journal and the final result (`protocol/result.json`); after every run the
campaign summary (`summary.json`, or `analysis-v2/summary.json` for teams) is rewritten from an independent replay.
Runs are executed one after another, and an interrupted campaign continues from its saved answers when the same
command is run again. With the paper's settings a run on one map costs USD 0.14 to 0.36 as reported by OpenRouter
and takes 5 to 10 minutes of search; on two floors it costs up to USD 0.70 and takes up to 20 minutes.

The constants of the theory (exact block minima, the realized fraction Q of the corrected improvement, correction
errors, the round-by-round bound) are computed from a team campaign by

```bash
python routing/analysis/theory/audit.py --root runs/map1-team --out runs/map1-team/theory
```

## Change the experiment

* **Prompts:** edit `prompt_team.py`, `prompt_single.py` or `two_floors.py` and check the result with `show_prompt.py`.
* **Model or sampling:** edit `settings` in a campaign file (`model`, `provider`, `reasoning_effort`, `temperature`,
  `top_p`), or copy it to a new campaign file.
* **Protocol:** the private horizon `team_H`, the round budget, the patience, the merge rule (`merge`:
  `best_single`, `best_subset` or `all`) and the correction (`cross_block_correction: false`) are campaign settings;
  the protocol itself is `robot_routing/team.py`.
* **Maps:** a map is `maps/<name>/instance.json` (grid, walls, robots with starts and goals, the agents' groups);
  point a campaign's `map` to a new folder. `reference.py` certifies an optimum for the audit.

## Recompute the results of the paper

```bash
python routing/analysis/run_all.py
```

reads the run records in `../data/routing-runs/` and the audited constants in `../data/routing-theory/`, and
recomputes in about two minutes, with no model calls, the statistics reported in the paper: per-campaign summaries
and permutation tests (`team-vs-single/`), the two-floor team composed from the runs of the two maps
(`two-floors/`), the comparison of the merge rules and the round-by-round bound with the realized and effective
numbers of colors (`merge-rules/`). The script lists the JSON files it writes.
