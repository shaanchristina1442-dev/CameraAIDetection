# Smoke test — clean clone verification

Date: 2026-11-03
Machine: Apple M5, 24 GB, macOS (Darwin 25.6.0), Python 3.14.7

Purpose: verify the README's "core pipeline works end-to-end" claim from a
clean clone. This is the part of **Task A** that does not require real footage.

## Result

The code is correct. Two packaging defects stopped a clean clone from running;
both are fixed. Every component that could be tested without an API key works.

## Defects found and fixed

| # | Defect | Effect on a clean clone | Fix |
|---|---|---|---|
| 1 | `ultralytics` missing from `requirements.txt` | `ImportError` on `pipeline/trigger.py:21`. Nothing runs. | Added `ultralytics>=8.3.0` |
| 2 | `sample_clips/synthetic_demo.mp4` committed as 0 bytes | `ValueError: Could not open video file`. README step 2 crashes. | Regenerated via `make_synthetic_clip.py` (466 KB, valid) |

Defect 2 masked defect 1 in the sense that either alone is enough to make the
documented quickstart fail. Both had to be fixed to get a passing run.

## Verified working

### Trigger layer (no API cost)

```
$ python -m pipeline.run_pipeline sample_clips/synthetic_demo.mp4 --dry-run
[pipeline] 1 candidate event(s) found
[pipeline] event 1: 00:00:03 - 00:00:07 (person present + high motion energy,
                                          3 people, motion 0.427)
```

Correct against known ground truth — the generator places people from 3s to 7s
and the trigger found exactly that window, no false events in the quiet
stretches either side. Scan took 4.2s for 10s of 15fps video, i.e. faster than
real time on CPU.

### Dashboard (Flask test client, all routes)

| Check | Result |
|---|---|
| `GET /` | 200, 7520 bytes |
| `GET /api/events` | 200, 4 events |
| `POST .../review` with `confirmed` | 200, persisted correctly |
| `POST .../review` with invalid status | 400 |
| `POST .../review` on unknown event | 404 |
| `GET /frames/<valid>` | 200, image served |
| `GET /frames/../../pipeline/config.py` | 404 — path traversal blocked |

Alert threshold logic correct on all four mock events: confidence 0.38
giveaway suppressed (below the 0.5 floor), 0.91 `nothing_notable` suppressed
(category excluded), 0.64 altercation and 0.82 concealment both raised.

### Dependency compatibility on Python 3.14

All install cleanly with wheels — no build-from-source:

```
anthropic==2.4.0        opencv-python==5.0.0.93   torch==2.15.0
Flask==3.1.3            python-dotenv==1.2.4      ultralytics==8.4.201
watchdog==7.0.0
```

Note `anthropic` resolves to 2.x. `anthropic.APIError` and
`anthropic.BadRequestError` both still exist, so `judge.py`'s exception
handling is valid on 2.x.

## Not verified

- **The judge layer.** Needs `ANTHROPIC_API_KEY` and costs money per event.
  Untested end-to-end. Note that Task B must land before this is worth
  testing, since the model/token-budget issue would make every call fail.
- **Accuracy on real footage.** The synthetic clip proves the pipeline runs;
  it says nothing about whether thresholds suit a real store. This is the
  remaining half of Task A and needs real footage.
- **Visual rendering of the dashboard.** Routes and data verified
  programmatically; not opened in a browser.

## Repo state after this run

- `requirements.txt` — fixed (1 line added)
- `sample_clips/synthetic_demo.mp4` — regenerated, now valid
- `events/events.jsonl` — contains 4 seeded mock events; clear with
  `python sample_clips/seed_mock_events.py --clear`
- `.claude/launch.json` — added (dashboard launcher)
- `.venv/`, `yolov8n.pt` — created, both gitignored

Also noted but not changed: `pipeline/__pycache__/judge.cpython-312.pyc` is
tracked in git despite `__pycache__/` being in `.gitignore` (committed before
the ignore rule). Harmless; remove with `git rm --cached`.
