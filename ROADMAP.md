# Project Roadmap

**Store Monitoring System** — an AI-assisted alerting tool that watches camera
footage and surfaces moments worth a human's attention.

---

## What we are building (and what we are not)

**We are building:** a system that watches live or recorded store cameras,
notices moments that might matter, describes what is visibly happening in plain
language, and puts them in a queue for a person to review and decide on.

**We are not building:** anything that accuses, identifies, or acts on its own.
Every output is "worth a look", never "confirmed". A person makes every decision.

This is not a disclaimer bolted on at the end — it is a design constraint that
shapes the schema, the UI copy, and what we deliberately refuse to build.

### Deliberately out of scope

| Excluded | Why |
|---|---|
| Face recognition / identity matching | Turns behavior monitoring into biometric data collection. Real statutory exposure (Illinois BIPA has a private right of action, Texas CUBI, GDPR Art. 9). Buys the reviewer nothing. |
| Automated action of any kind | No auto-notifying police, no auto-locking doors, no auto-flagging a person as banned. The human decides. |
| Cross-session person re-identification | Track IDs are temporary and discarded when a person leaves the frame. We never build a history of an individual. |
| Employee misconduct categories (until POS data exists) | Not visually distinguishable from a legitimate comp, discount, or return. False positives name a specific employee. See Task Z notes. |

### Non-code homework (the owner's responsibility, not the software's)

These are not engineering tasks, but the project is not "a success" without them:

- Customer and employee notice/signage where cameras operate, per local law
- A written retention policy (how long footage and alerts are kept) — Task T enforces it in code, but someone has to decide the number
- Deciding who is allowed to review alerts, and what they are allowed to do with a confirmed one
- Checking whether employee monitoring in your jurisdiction requires separate notice or consent

---

## How the system works today

Four stages, cheap to expensive. The design principle: use the free stage to
decide when to spend money on the expensive stage.

| Stage | File | Cost | Job |
|---|---|---|---|
| 1. Watch | `pipeline/trigger.py` | Free | Pixel-change detection + YOLO person detection. Answers "might something be happening?" |
| 2. Judge | `pipeline/judge.py` | ~1c/event | Sends frames to Claude. Returns category, confidence, plain-language description. |
| 3. Record | `pipeline/run_pipeline.py` | Free | Saves frames, appends to the event log. |
| 4. Show | `dashboard/app.py` | Free | Local web dashboard. Lists events, Confirm/Dismiss buttons. |

`pipeline/watcher.py` automates stages 1–3 for any video dropped into `footage/`.

---

## Target architecture (where we are going)

```
 CAMERAS                  WORKERS                      STORAGE        INTERFACE
 ┌──────────┐    ┌──────────────────────┐    ┌──────────┐   ┌──────────────┐
 │ RTSP cam │───►│ Capture (1 per cam)  │    │          │   │  Dashboard   │
 │ RTSP cam │───►│  tight loop, never   │    │  SQLite  │◄──│  (review UI) │
 │ RTSP cam │───►│  blocks; keeps last  │    │          │   └──────────────┘
 └──────────┘    │  30s in memory       │    │  events  │   ┌──────────────┐
                 └──────────┬───────────┘    │  cameras │◄──│  PWA / phone │
                            │ frames         │  zones   │   └──────────────┘
                            ▼                │  reviews │          ▲
                 ┌──────────────────────┐    │  users   │          │
                 │ Trigger worker       │    │  devices │    ┌───────────┐
                 │  motion + YOLO +     │    └─────▲────┘    │ Web Push  │
                 │  tracking + zones    │          │         └───────────┘
                 └──────────┬───────────┘          │               ▲
                            │ candidate events     │               │
                            ▼                      │               │
                 ┌──────────────────────┐          │               │
                 │ Judge worker pool    │──────────┘               │
                 │  dedup, cooldown,    │──────────────────────────┘
                 │  Claude, confidence  │   high-confidence alerts
                 └──────────────────────┘
```

The critical change from today: **capture never waits for Claude.** A video file
is patient; a camera is not. Pieces run concurrently and hand work to each other
through queues.

---

## Phases at a glance

| Phase | Tasks | Theme | Rough effort |
|---|---|---|---|
| 0 | A–E | Make what exists correct | 2–3 days |
| 1 | F–G | Real database | 2 days |
| 2 | H–L | Restructure for concurrency | 4–6 days |
| 3 | M–O | Live RTSP cameras | 4–5 days |
| 4 | P–Q | Camera management + zones | 4–5 days |
| 5 | R–T | Auth + hardening + retention | 3–4 days |
| 6 | U–W | Push notifications | 3–4 days |
| 7 | X | Mobile PWA | 3–4 days |
| 8 | Y–Z | Measurement + ship | 4–5 days |

Estimates assume one developer working with AI assistance, and are deliberately
rough. Phase 3 is the one most likely to overrun — real cameras misbehave in
ways you cannot predict from a spec.

**Ordering rule:** Phases 0–2 must happen in order. Phase 2 before Phase 3 is
non-negotiable: three later features live inside the structure Phase 2 creates,
so doing them first means building them twice. Phases 4–7 are more flexible.

---

# The Tasks: A to Z

Each task: what it is, why it matters, where it lives, and how you know it is done.

---

## Phase 0 — Make what exists correct

### A. Set up the environment and establish a baseline

**What:** Get the project running end to end on your machine, then run it on
real store footage and write down exactly how it performs today — how many
events it flagged, how many were real, how many it missed, what it cost.

**Why:** Every later task claims to improve accuracy. Without a "before"
number, none of those claims are checkable. This is the single most skipped
step in projects like this, and skipping it means you will never know whether
your changes helped.

**Where:** `.venv`, `.env`, `footage/`, plus a new `benchmarks/baseline.md`

**Done when:** `python -m pipeline.run_pipeline footage/<real_clip>.mp4` completes
on real footage, and `benchmarks/baseline.md` records: clip length, events
flagged, how many you judged correct on review, obvious misses, total API cost,
and wall-clock scan time.

**Note:** You need real footage, not `sample_clips/synthetic_demo.mp4`. Even a
few minutes containing one known incident is enough to be useful. Several
later tasks (C, E, Y) cannot be evaluated without it.

---

### B. Fix the judging layer

**What:** Three changes that must land together:
1. Model `claude-sonnet-4-6` → `claude-sonnet-5-5` (newer and cheaper: $2/$10 per
   million tokens vs $3/$15)
2. Raise `max_tokens` from 450 to ~2000 and set `output_config={"effort": "low"}`
3. Replace hand-rolled JSON parsing with structured outputs
   (`output_config={"format": {...}}`), and check `stop_reason` before reading
   the response

**Why:** The newer model thinks before answering, and thinking consumes the same
token budget as the answer. At 450 tokens it would spend the whole budget
thinking and return a truncated response — which your parser cannot read, so
*every event* would log as `json_parse_failed`. Changing the model name alone
looks like a total system failure. Structured outputs then remove the fragile
backtick-stripping workaround entirely, and the `stop_reason` check catches
safety refusals (which arrive as HTTP 200, so they currently masquerade as
parse bugs).

**Where:** `pipeline/config.py:69`, `pipeline/judge.py:107-145`

**Done when:** A full run on the baseline clip produces zero parse errors, and
`judge.py` contains no string surgery on the response.

---

### C. Send person-crops instead of whole frames

**What:** Pass YOLO's bounding boxes through `TriggerEvent` to the judge, and
send a padded crop around the person rather than the whole downscaled frame.

**Why:** This is the highest-leverage accuracy change available. Frames are
currently shrunk to 640px wide, so on a wide store shot a person is ~80px tall
and a hand entering a pocket is a dozen pixels — genuinely invisible to the
model. `_detect_objects` already computes exact person boxes and then discards
them. Cropping gives a far clearer picture at identical token cost.

**Where:** `pipeline/trigger.py` (add `detections` to `TriggerEvent`),
`pipeline/judge.py` (`_frame_to_base64` → crop-aware)

**Done when:** Saved event frames are person-centered crops, and a re-run on the
baseline clip shows measurably more specific descriptions.

**Open question:** Multiple people in frame — one crop per person, or one wider
crop covering all of them? Resolve by testing on real footage; context between
two people matters for altercations but hurts for concealment.

---

### D. Clean up and fix the small bugs

**What:**
- Delete `recognize_face` stub (`trigger.py:184`)
- Delete dead `define_issues` (`trigger.py:175`) and the unreachable
  `return issue` (`trigger.py:201`)
- Replace `event_id` with a UUID (`run_pipeline.py:74`)

**Why:** The face stub invites a future contributor to implement the one thing
we have decided not to build. The event ID is `{timestamp}_{idx}`, which
collides when two clips process within the same second — exactly what the
watcher does on a batch of short clips. Collisions mean overwritten frame
images and a Confirm click landing on the wrong event.

**Where:** `pipeline/trigger.py`, `pipeline/run_pipeline.py`

**Done when:** Two clips processed in the same second produce distinct IDs and
distinct image files.

---

### E. Re-measure against the baseline

**What:** Re-run the same footage from Task A and compare.

**Why:** Closes the loop. If B and C did not improve things, better to know now
than after building six more phases on top.

**Where:** `benchmarks/after-phase-0.md`

**Done when:** A written side-by-side comparison exists. If accuracy did not
improve, investigate before proceeding.

---

## Phase 1 — Real database

### F. Move to SQLite

**What:** Replace `events/events.jsonl` and `events/reviews.json` with a SQLite
database. Tables: `events`, `cameras`, `zones`, `reviews`, `users`, `devices`.

**Why:** Append-only JSON files cannot survive multiple camera workers writing
at once — which Phase 2 creates. The dashboard already works around this by
keeping reviews in a separate file to avoid rewriting the log mid-append. That
workaround does not scale, and you cannot query a JSONL file for "high-confidence
theft events on camera 3 last Tuesday".

**Where:** new `storage/schema.sql`, `storage/db.py`

**Done when:** Schema is created, indexed on `(camera_id, created_at)` and
`(review_status)`, and concurrent writes from two processes do not corrupt it.

---

### G. Shared data-access layer and migration

**What:** One module both the pipeline and the dashboard import for all reads
and writes. Plus a one-off script migrating existing `events.jsonl` rows in.

**Why:** If the pipeline and dashboard each write SQL their own way, they will
drift. A single module is also where alert-suppression logic (Task K) naturally
lives.

**Where:** `storage/repository.py`, `scripts/migrate_jsonl_to_sqlite.py`

**Done when:** No SQL outside `storage/`, dashboard reads only through the
repository, and existing events appear in the dashboard after migration.

---

## Phase 2 — Restructure for concurrency

### H. Add object tracking

**What:** Add ByteTrack (built into ultralytics) so each person gets a stable ID
across frames while visible.

**Why:** Prerequisite for de-duplication (Task K). Without it you cannot tell
"one person loitering for two minutes" from "forty separate events", and that
distinction is what stands between a useful system and one the owner mutes.

**Where:** `pipeline/trigger.py` (`model.track()` instead of `model.predict()`)

**Done when:** A person walking across frame keeps one ID; IDs are discarded when
they leave, never persisted.

---

### I. Split into capture / trigger / judge workers

**What:** Break the synchronous loop into three concurrent pieces connected by
queues: capture (never blocks), trigger (motion + YOLO + tracking), judge
(Claude calls, pooled).

**Why:** The core restructure. Today the loop waits for Claude; a camera does
not wait, so frames pile up or get dropped and latency grows all shift. Also
means one slow Claude call no longer stalls detection on every camera.

**Where:** new `pipeline/workers/{capture,trigger,judge}.py`, `pipeline/queues.py`

**Done when:** A file-based run still produces identical results, but trigger
and judge run concurrently, and an artificially slow judge call does not slow
capture.

---

### J. Rolling pre-roll buffer

**What:** Capture keeps the last ~30 seconds of frames in memory. When a trigger
fires, reach backwards into it.

**Why:** Today `CLIP_PADDING_SECONDS` gets pre-event context free because you can
seek in a file. Live, you only have it if you deliberately kept it — and for
concealment the informative footage is almost always *before* the motion spike.

**Where:** `pipeline/workers/capture.py` (ring buffer), `pipeline/config.py`

**Done when:** A triggered event includes frames from before the trigger moment,
and memory use stays bounded with several cameras running.

---

### K. Alert de-duplication and cooldowns

**What:** Suppression rules: one alert per track ID per cooldown window, a
confidence floor separating "log it" from "alert on it", and time-windowed
aggregation.

**Why:** **Alert fatigue is the failure mode that kills systems like this** —
not accuracy. Thirty pushes a shift and the owner mutes it, and a muted system
is worth exactly zero no matter how good the detection. Everything logs; only
high-confidence events alert.

**Where:** `pipeline/workers/judge.py`, `storage/repository.py`,
`pipeline/config.py`

**Done when:** One person loitering produces one alert, not forty, and the
dashboard still shows every underlying event.

---

### L. Process supervision and graceful shutdown

**What:** One entry point that starts all workers, restarts a crashed one, and
shuts down cleanly on Ctrl-C without losing queued events.

**Why:** This becomes a thing that runs for weeks. A camera worker crashing at
3am must not silently stop monitoring that camera.

**Where:** new `pipeline/supervisor.py`

**Done when:** Killing a worker gets it restarted and logged; Ctrl-C drains
queues before exiting.

---

## Phase 3 — Live cameras

### M. RTSP ingestion with reconnect

**What:** Capture reads from an RTSP URL, with exponential-backoff reconnect,
stall detection (stream open but no frames), and logging of every drop.

**Why:** This is the feature you actually want, and most of the work is not
"connect" — it is "stay connected". Cameras drop constantly: network blips,
power cycles, DVR reboots, bandwidth saturation. A naive implementation works
for an hour and fails overnight.

**Where:** `pipeline/workers/capture.py`, `pipeline/sources/rtsp.py`

**Done when:** Pulling the network cable and plugging it back in results in
automatic recovery with no restart, and the gap is recorded.

---

### N. Multi-camera support

**What:** Run several cameras at once, each with its own capture worker and
config, sharing the trigger and judge pools.

**Why:** A real store has more than one camera, and per-camera thresholds matter
— a door camera and a stockroom camera need different sensitivity.

**Where:** `pipeline/supervisor.py`, `storage/schema.sql` (cameras table)

**Done when:** Three streams run concurrently with per-camera settings, events
are correctly attributed, and one failing camera does not affect the others.

---

### O. Per-camera health monitoring

**What:** Track last-frame time, reconnect count, FPS, and queue depth per
camera. Surface it in the dashboard.

**Why:** A camera that silently stopped is worse than no camera, because you
believe you have coverage. This is also the most likely real-world failure.

**Where:** `storage/repository.py`, `dashboard/app.py`, `dashboard/templates/`

**Done when:** The dashboard shows live status per camera and flags any camera
with no frames for 60+ seconds.

---

## Phase 4 — Camera management and zones

### P. Camera management UI

**What:** Add, name, edit, and remove cameras from the dashboard. RTSP
credentials encrypted at rest, never rendered back to the browser.

**Why:** Editing a Python config file to add a camera is not a product. The
encryption matters: RTSP URLs contain passwords, and a camera password often
unlocks the whole DVR.

**Where:** `dashboard/app.py`, new `dashboard/templates/cameras.html`,
`storage/repository.py`

**Done when:** A camera can be added from the UI and starts being monitored
without a restart, and its password is not retrievable through the web UI.

---

### Q. Detection zones and per-zone rules

**What:** Draw polygons on a camera snapshot to define zones. Per-zone rules:
sensitivity, which categories apply, and active hours.

**Why:** Sharply cuts false positives and enables rules impossible globally —
"any person in the stockroom after 8pm" is a strong signal; "any person in the
aisle" is noise. Also lets you exclude the ceiling fan and the TV screen that
currently trigger motion.

**Where:** `pipeline/workers/trigger.py`, `dashboard/templates/zones.html`,
`storage/schema.sql`

**Done when:** A zone drawn in the UI changes triggering behavior, and an
after-hours rule fires on schedule.

---

## Phase 5 — Auth and hardening

### R. User authentication

**What:** Login with properly hashed passwords, server-side sessions, CSRF
protection on review actions.

**Why:** Prerequisite for push notifications — a notification needs a user to
belong to. Also: this dashboard shows footage of identifiable people, which is
exactly the data that should not be open on a network.

**Where:** `dashboard/auth.py`, `dashboard/app.py`, `storage/schema.sql`

**Done when:** Unauthenticated requests to any route except login are rejected,
and passwords are stored only as salted hashes.

---

### S. Production hardening

**What:** Remove `app.run(debug=True)`, serve via a real WSGI server (gunicorn or
waitress), HTTPS, security headers, rate-limited login.

**Why:** `debug=True` exposes the Werkzeug console — remote code execution for
anyone who reaches the port. Harmless on localhost, catastrophic the moment this
is reachable from a phone. Phase 6 makes it reachable from a phone.

**Where:** `dashboard/app.py:89`, new `deploy/` config

**Done when:** The debug server is gone, HTTPS serves the dashboard, and a
security scan finds no critical issues.

---

### T. Audit log and retention

**What:** Record who reviewed what and when, immutably. A scheduled job purges
footage and events past the retention window, keeping escalated ones.

**Why:** The audit log is what makes "human-in-the-loop" verifiable rather than
asserted — if a confirmed alert ever leads to a consequence for a person, the
record of who decided matters. Auto-purge keeps a written retention policy
honest instead of aspirational, and limits the damage if the system is ever
breached.

**Where:** `storage/schema.sql`, `storage/repository.py`, new
`scripts/purge_expired.py`

**Done when:** Every review action is attributed and immutable, and the purge job
removes expired events and their image files while preserving escalated ones.

---

## Phase 6 — Push notifications

### U. Web Push backend

**What:** VAPID key generation, subscription storage, and sending via
`pywebpush`.

**Why:** Reaches desktop and iOS 16.4+ (installed to home screen) with no
Firebase dependency and no third-party SDK in the backend. FCM earns its keep
for fan-out to thousands of devices and delivery analytics; for a single-owner
tool it is a dependency without a payoff.

**Where:** new `notifications/webpush.py`, `storage/schema.sql` (devices)

**Done when:** A test notification arrives on a real phone and a real desktop.

---

### V. Device registration and notification rules

**What:** Register/unregister devices from the UI. Per-user rules: confidence
floor, category filter, quiet hours, per-camera mute.

**Why:** Task K suppressed duplicates; this gives the owner control over the
rest. The single most important setting is the confidence floor — the difference
between a tool they trust and one they mute.

**Where:** `notifications/rules.py`, `dashboard/templates/settings.html`

**Done when:** A device can be added and removed, and an event below the user's
floor logs without notifying.

---

### W. Deep-link from notification to event

**What:** Tapping a notification opens that specific event's review screen, with
the snapshot, description, confidence, and Confirm/Dismiss.

**Why:** The whole point. A notification that drops you on a home screen to hunt
for the event costs enough friction that review stops happening.

**Where:** `notifications/webpush.py` (payload), PWA service worker

**Done when:** Tapping a notification on a locked phone lands on the right
event, and Confirm works from there.

---

## Phase 7 — Mobile

### X. Progressive Web App

**What:** Web manifest, service worker, installable to home screen, mobile-first
review screen, and review actions that queue offline and sync when connectivity
returns.

**Why:** Installation is not cosmetic on iOS — Web Push only works for installed
PWAs. The offline queue matters because stockrooms and back offices have bad
signal, and a Dismiss that silently fails teaches the owner not to trust it.

**Where:** `dashboard/static/manifest.json`, `dashboard/static/sw.js`,
mobile templates

**Done when:** Installable on iOS and Android, notifications arrive while
closed, and a review made offline syncs on reconnect.

---

## Phase 8 — Measurement and ship

### Y. Labeled eval set and threshold tuning

**What:** Build a set of real clips with known, hand-labeled ground truth
(including clips where nothing happens). Measure precision and recall. Tune
`MOTION_THRESHOLD`, `ALERT_CONFIDENCE_FLOOR`, and the prompt against it. Wire
confirm/dismiss data into a live precision metric.

**Why:** Up to here, "is it working?" has been a judgment call. This replaces
opinion with a number, and makes every future change checkable. The
confirm/dismiss feedback is the highest-value data the system produces — it is
ground truth, generated free, by the person using it.

**Where:** new `benchmarks/eval_set/`, `benchmarks/run_eval.py`,
`dashboard/templates/metrics.html`

**Done when:** `python benchmarks/run_eval.py` reports precision and recall, and
the dashboard shows a rolling confirm rate.

**Include clips where nothing happens.** A system measured only on real
incidents looks excellent while crying wolf constantly.

---

### Z. Deployment, documentation, and fairness monitoring

**What:** Deployment runbook (install, start on boot, logs, backup, restore),
updated README, a troubleshooting guide, and a flag-rate monitor that reports
alert rates by camera, zone, and hour.

**Why:** The runbook is what makes this survivable by someone other than the
person who wrote it. The fairness monitor is the other half of the
human-in-the-loop promise: the system should be checked for *who* it flags
disproportionately, and the honest version of that check is monitoring flag
rates rather than asserting neutrality. No demographic attributes are ever
collected or inferred — the monitor watches rates by camera, zone, and time,
which is where skew actually shows up.

**Where:** `docs/runbook.md`, `docs/troubleshooting.md`, `README.md`,
`dashboard/templates/metrics.html`

**Done when:** A person who did not build this can install and run it from the
runbook alone, and the metrics page shows flag rates sliced by camera, zone, and
hour.

---

## Cross-cutting: POS integration (unscheduled)

Not lettered because it is gated on something outside the codebase: access to
your point-of-sale system's transaction data.

Correlating register events with camera events unlocks the two categories
currently unreliable by construction:

- `item_removal_no_transaction` — becomes checkable rather than guessed
- `employee_unauthorized_giveaway` — currently indistinguishable from a
  legitimate comp, discount, or return

**Until this exists:** keep both employee categories dashboard-only with no push
notifications. Unlike an anonymous shopper flag, a false positive here names a
specific employee with their shift on record. Worth doing properly or not at all.

Slot it after Phase 3 if POS access arrives early; it depends on the database
(Phase 1) but nothing later.

---

## Definition of success

The project is a success when:

1. It runs unattended for a week on real cameras without intervention
2. The owner reviews alerts voluntarily, because the signal-to-noise is worth it
3. Precision is measured, not guessed, and tracked over time
4. A confirmed alert has an audit trail showing who decided and when
5. Someone other than the builder can install and operate it from the runbook
6. It has never accused anyone of anything — it has only ever said "worth a look"

The second one is the real test. A technically excellent system that gets muted
in week two has failed. Everything in Task K and Task V exists to prevent that.
