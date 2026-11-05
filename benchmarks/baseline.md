# Baseline — first real footage

Date: 2026-11-04
Config: `claude-opus-5-5`, 3000 tokens, effort low, person-crops on,
`MOTION_THRESHOLD=0.02`, `FRAME_SKIP=5`, `yolov8n.pt`

**This is not yet a usable baseline.** Two clips, both with defects that
invalidate parts of the measurement. Recorded here so the next attempt has
something to beat and does not repeat the same mistakes.

## Result: 1 of 2 correct

| Clip | Expected | Judged | Conf | |
|---|---|---|---|---|
| `browsing.mov` | `nothing_notable` | `nothing_notable` | 0.55 | correct |
| `concealment_pocket.mov` | `theft_concealment` | `nothing_notable` | 0.60 | **miss** |

Both triggered exactly one event, both recommended human review.

## The miss is a data defect, not a model failure

Claude's own words on the concealment clip: *"the only notable action is a
person picking up a small item from a desk and walking away, done openly
with no visible concealment."*

That description is accurate. Watching the frames, the item is picked up and
carried visibly past the camera. **The clip does not depict concealment**, so
the label is wrong rather than the judgment. A model that flagged it would be
guessing at intent from an action that looks ordinary — exactly the failure
mode this system is supposed to avoid.

Second factor: the setting is a home office, and Claude reasoned *"no
register, merchandise displays or checkout area are visible, so this does not
look like a retail sales floor."* The categories are retail-specific, so a
non-retail scene makes the model reasonably discount all of them. Staged
footage needs to read as a shop, or the retail categories cannot be exercised.

## What the detection layer did well

- YOLO: 0.78–0.96 confidence on people across both clips, no false boxes
- Action tracking was accurate: *"walks to the desk, reaches toward a small
  white item, picks it up, and turns toward the camera holding it"*
- Confidence was honest (0.55, 0.60) and both were marked for human review —
  the right posture on ambiguous input
- 1080p H.264 from an iPhone decodes and runs without conversion

## What invalidated the rest

**Handheld camera.** Motion energy 0.579 and 0.751, against 0.427 for a
synthetic clip with a bus sliding across the whole frame. Frame-differencing
compares consecutive frames, so a moving camera changes every pixel and the
score stops meaning anything. Nothing can be learned about `MOTION_THRESHOLD`
from these clips. The camera must be stationary, not merely well-placed.

**Hand over the lens.** Both clips end with a forearm reaching in to stop the
recording, which consumes one of the five frames sent for judgment. Claude
noted it unprompted.

## For the next attempt

1. Phone propped against something solid. Nobody touches it while recording.
2. Start recording, walk away, perform, walk back, stop. No hand in frame.
3. Make the scene read as retail: several items lined up on a table or shelf
   so they look like stock rather than someone's desk.
4. Perform concealment properly: the item must go *into* a pocket or bag and
   stop being visible. If it stays in view, the clip is `browsing`.
5. Record all six scenarios. `own_bag` paired with `concealment_pocket` is
   the pair that matters; neither is informative alone.

---

# Real CCTV baseline — 2026-11-04

First measurement on genuine fixed-camera surveillance footage.
`footage/cafe_cctv.mp4`, 4m12s, an outdoor office/café terrace.

Source was a phone screen-recording of a camera app, portrait and **rotated
90°**. YOLO found 1 person across 8 sample frames as-is, 12 rotated — so
the clip was unusable until normalised. `sample_clips/prep_footage.py`
(new) does this, with `--detect-rotation` to choose automatically.

## Headline numbers

| | |
|---|---|
| Scan speed | 252s of video in 23s — **11× real time** on CPU |
| Events found | 6 |
| Event rate | **86 events/hour** |
| Alerts raised | 1 |
| Alert rate | **14 alerts/hour** |
| API cost | ~$0.94/hour, **~$11 per camera per 12-hour day** |

## Per-event results

| # | Window | People | Motion | Judged | Conf | Alert |
|---|---|---|---|---|---|---|
| 1 | 43–57s | 2 | 0.52 | `altercation_or_aggression` | 0.50 | **yes** |
| 2 | 71–95s | 2 | 0.22 | `nothing_notable` | 0.60 | no |
| 3 | 112–122s | 3 | 0.22 | `altercation_or_aggression` | 0.40 | no |
| 4 | 134–140s | 1 | 0.20 | `nothing_notable` | 0.55 | no |
| 5 | 175–231s | 5 | 0.52 | `nothing_notable` | 0.85 | no |
| 6 | 240–249s | 4 | 0.73 | `nothing_notable` | 0.85 | no |

No ground truth for this clip, so these are not scored. The two
`altercation` calls are near-certainly false positives — it is a café
terrace, not a fight.

## What worked

**The trigger discriminates on fixed-camera footage.** Six separate events
with distinct windows, rather than the one all-encompassing event the
handheld clips produced. `MOTION_THRESHOLD=0.02` is defensible here;
motion ranged 0.20–0.73 across events.

**The confidence floor did its job.** Event 3 (`altercation`, 0.40) was
logged but not alerted; event 1 (0.50) was. That is `ALERT_CONFIDENCE_FLOOR`
working exactly as designed — the mechanism in task K, demonstrated on real
footage.

**The model hedges honestly.** Every flag carried explicit uncertainty:
*"no clear strike or physical contact is visible"*, *"could equally be
benign, such as adjusting clothing, handing something over, or animated
conversation"*. It never claimed certainty it did not have. For a
human-in-the-loop tool this matters more than raw accuracy — a reviewer can
calibrate against honest confidence.

## What this exposes

**14 false alerts per hour is unusable.** At that rate an owner mutes the
system within a day, and a muted system is worth nothing however good the
detection. This is the first hard number for the alert-fatigue problem that
tasks K and V exist to solve.

Cheapest lever: raise `ALERT_CONFIDENCE_FLOOR` from 0.5 to 0.6. On this
clip that removes the only alert. Needs true positives before it can be
tuned properly — raising the floor also suppresses real events.

**~$11 per camera per 12-hour day** at 86 events/hour. Three cameras is
~$1,000/month in API cost alone. Trigger precision is therefore a cost
control, not only a quality one: every junk event is a paid call. Zone
rules (task Q) and de-duplication (task K) pay for themselves here.

**The model noticed the footage is non-contiguous.** On event 5 it read the
on-screen timestamps and observed the frames *"span roughly 12:13 to
12:43"* — 30 minutes of camera time inside a 56-second event, because the
recording is someone scrubbing through playback. The pipeline treats it as
continuous video. Worth knowing before trusting any timing from this clip.

**Every judgment noted the setting is not retail.** The categories are
retail-specific and this is an office terrace, so the model repeatedly
discounted them. Real store footage would exercise them properly.

## Next

1. Real store footage, fixed camera, not a screen recording of playback
2. Ground truth labels, so these numbers become precision and recall
3. Then tune `ALERT_CONFIDENCE_FLOOR` against it — currently 0.5, and
   0.6 looks better on this evidence alone
