# store-monitor-prototype

Prototype retail store monitoring system that watches for security events in video clips, classifies them with Claude, and surfaces results on a local dashboard.

## Project layout

```
pipeline/
  config.py          # thresholds, paths, Claude model; loads .env
  trigger.py         # YOLO + motion scan: finds candidate events in a video
  judge.py           # sends event frames to Claude, returns a classification
  run_pipeline.py    # one clip: trigger -> judge -> events/events.jsonl
  watcher.py         # watches footage/ and runs the pipeline on each new clip

dashboard/
  app.py             # Flask app serving the event log + review actions
  templates/
    index.html       # event feed UI (auto-refreshes every 5s)

sample_clips/
  make_synthetic_clip.py   # generate a test .mp4 with people walking through
  seed_mock_events.py      # write fake rows into events.jsonl (no YOLO/Claude)

footage/             # drop store footage here (git-ignored)
events/
  events.jsonl       # append-only log of judged events
  clips/             # saved frames for each event (git-ignored)
  reviews.json       # reviewer confirm/dismiss decisions (git-ignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
```

On Ubuntu/WSL, if `venv` fails run `sudo apt install python3.12-venv` first.

## Running

Run everything from the project root, with the venv active. Use `python -m`
(not `python pipeline/xyz.py`) because the pipeline modules use package imports.

```bash
# 1. Check the dashboard works with fake data (no API calls)
python sample_clips/seed_mock_events.py
python -m dashboard.app                     # -> http://127.0.0.1:5000

# 2. Tune the trigger on your footage first - YOLO only, no Claude cost
python -m pipeline.run_pipeline footage/store_cam.mp4 --dry-run

# 3. Full run on one clip (calls Claude for each event)
python -m pipeline.run_pipeline footage/store_cam.mp4

# 4. Or leave the watcher running and copy clips into footage/
python -m pipeline.watcher
```

Remove the mock events again with `python sample_clips/seed_mock_events.py --clear`.

## Testing on store footage

- Export clips as `.mp4`/`.avi`/`.mov`/`.mkv`. Proprietary DVR formats (`.dav`,
  `.264`, etc.) must be converted first, e.g.
  `ffmpeg -i input.dav -c:v libx264 footage/out.mp4`.
- Start with a short clip (a few minutes) and `--dry-run`. Each candidate event
  costs one Claude call with 5 frames.
- Too many events (fans, screens, lighting flicker): raise `MOTION_THRESHOLD`.
  Missing people in wide shots: lower it, or set `YOLO_MODEL=yolov8s.pt`.
- Scanning is CPU-bound. Quiet footage is fast because YOLO only runs when
  something moves; busy footage runs at roughly real time or slower on a laptop
  CPU. Raise `FRAME_SKIP` (e.g. 10) to go faster.
- All of these can be set in `.env` without editing code.

## How it works

1. `trigger.py` checks every `FRAME_SKIP`th frame for motion. When something
   moves it runs YOLO; a person plus motion opens an event, which closes after
   `MIN_SECONDS_BETWEEN_EVENTS` of quiet (or after `MAX_EVENT_SECONDS`).
2. `run_pipeline.py` picks 5 frames spread across each event, saves them, and
   calls `judge.py`.
3. `judge.py` asks Claude to describe and classify the frames into one of the
   categories in `config.DETECTION_CATEGORIES`, with a confidence score.
4. Events that aren't `nothing_notable` and meet `ALERT_CONFIDENCE_FLOOR` are
   marked as alerts. The dashboard shows them for a human to confirm or dismiss.

This is a human-in-the-loop review tool: every result is "worth a look", not a
confirmed incident.
