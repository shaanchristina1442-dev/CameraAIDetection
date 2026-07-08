# store-monitor-prototype

Prototype retail store monitoring system that watches for security events in video clips, classifies them with Claude, and surfaces results on a local dashboard.

## Project layout

```
pipeline/
  config.py          # env vars, paths, thresholds
  trigger.py         # watches for new clips and kicks off the pipeline
  run_pipeline.py    # orchestrates clip → judge → events.jsonl
  judge.py           # sends clip frames to Claude, returns event classification

dashboard/
  app.py             # Flask app serving the event log
  templates/
    index.html       # live event feed UI

sample_clips/
  make_synthetic_clip.py   # generate a test .mp4 with baked-in events
  seed_mock_events.py      # write synthetic rows into events/events.jsonl

events/
  events.jsonl       # append-only log of detected events
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
```

## Running

```bash
# Generate a test clip
python sample_clips/make_synthetic_clip.py

# Seed mock events (skips the pipeline)
python sample_clips/seed_mock_events.py

# Start the pipeline watcher
python pipeline/trigger.py

# Start the dashboard (separate terminal)
python dashboard/app.py
# → http://localhost:5000
```

## How it works

1. `trigger.py` watches a clip directory for new `.mp4` files.
2. Each new clip is passed to `run_pipeline.py`, which samples frames and calls `judge.py`.
3. `judge.py` sends frames to Claude with a classification prompt; the result is appended to `events/events.jsonl`.
4. The Flask dashboard polls `events.jsonl` and renders the event feed in real time.
