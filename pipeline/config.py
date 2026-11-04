"""
Central configuration for the store monitoring prototype.
 
IMPORTANT CONTEXT FOR ANYONE EXTENDING THIS:
This system is designed as a HUMAN-IN-THE-LOOP ALERTING TOOL, not an
autonomous accusation system. It surfaces moments worth a human's
attention. It does not label anyone a "thief" or take any automated
action. All outputs should be treated as "worth reviewing", not "confirmed".
"""
import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pull ANTHROPIC_API_KEY (and any overrides) from <project>/.env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
 
# --- Trigger layer (YOLO) settings ---
 
# Pretrained YOLO weights (downloaded automatically on first run).
# yolov8n is fastest; yolov8s/m are more accurate on small, distant people.
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")

# How often to run object detection (every Nth frame). Lower = more
# sensitive / slower. 5 on 30fps store footage = 6 checks per second,
# which is plenty to catch a person and keeps CPU scanning near real time.
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 5))
 
# Confidence threshold for YOLO person/object detections
YOLO_CONF_THRESHOLD = 0.45
 
# Classes from the COCO dataset (what the pretrained YOLO model knows)
# that are relevant triggers. "person" is the main one we care about;
# others help us reason about hand-near-object events.
RELEVANT_CLASSES = ["person", "backpack", "handbag", "suitcase", "bottle", "cell phone"]
 
# Motion energy = fraction of the frame (0-1) whose pixels changed by more
# than MOTION_PIXEL_DELTA (0-255) since the last checked frame. 0.02 means
# "2% of the picture moved". Raise it if fans/screens/lighting cause false
# triggers; lower it for wide shots where people look small.
MOTION_THRESHOLD = float(os.environ.get("MOTION_THRESHOLD", 0.02))
MOTION_PIXEL_DELTA = 25

# Events shorter than this are dropped as noise (someone walking past)
MIN_EVENT_SECONDS = 1.0

# Continuous activity longer than this is split into separate events so
# each one is short enough for Claude (and a reviewer) to judge
MAX_EVENT_SECONDS = 60

# Upper bound on frames held in memory per event while scanning
MAX_FRAMES_KEPT_PER_EVENT = 60
 
# An event only ends after this many seconds of no activity, so a short
# pause doesn't split one moment into several near-duplicate events
MIN_SECONDS_BETWEEN_EVENTS = 4
 
# How many seconds of context to grab around a trigger (before/after)
CLIP_PADDING_SECONDS = 2
 
# How many frames to sample from a flagged clip to send to the
# judgment layer (more frames = better judgment, more $ per event)
FRAMES_PER_JUDGMENT = 5
 
 
# --- Judgment layer (Claude vision) settings ---
 
CLAUDE_MODEL = "claude-opus-5-5"

# Thinking is always on for this model and shares the response budget, so
# max_tokens must leave room for it or the JSON comes back truncated. This is
# a cap, not a spend: unused budget costs nothing, so it is set generously.
JUDGE_MAX_TOKENS = 3000

# How hard the model thinks before answering: low | medium | high | xhigh | max.
# This model defaults to "medium", so it is set explicitly. "low" suits a
# 7-way classification; raise it if ambiguous frames are being mislabelled,
# and re-measure against benchmarks/ rather than guessing.
JUDGE_EFFORT = "low"


# --- Frame cropping (what the judgment layer actually sees) ---

# Send a crop around the detected people instead of the whole downscaled frame.
# On a wide store shot a person is a small part of the picture, and the detail
# that decides the category (a hand entering a pocket) is a few pixels. YOLO
# already knows where the people are, so spend the resolution there.
CROP_TO_PEOPLE = True

# Padding around the people, as a fraction of the box size. Context matters:
# the shelf being reached into is outside the person's own bounding box.
CROP_PADDING = 0.35

# Never send a crop smaller than this fraction of the frame; a tiny sliver
# loses the surroundings a reviewer needs to make sense of the scene.
CROP_MIN_FRACTION = 0.25

# If the people already fill this much of the frame, cropping gains nothing.
CROP_SKIP_ABOVE = 0.8
 
# The categories we ask Claude to classify flagged events into.
# Keep this list as the single source of truth - the prompt builder
# in judge.py reads from this.
DETECTION_CATEGORIES = {
    "theft_concealment": (
        "A person appears to conceal store merchandise on their body, in "
        "clothing, or in a bag in a way that looks deliberately hidden "
        "rather than a normal shopping motion."
    ),
    "item_removal_no_transaction": (
        "A person appears to leave a checkout/register area or exit a "
        "frame with merchandise without any visible scanning, payment, "
        "or bagging interaction."
    ),
    "property_damage": (
        "Someone appears to be damaging, breaking, or vandalizing store "
        "property, fixtures, or merchandise."
    ),
    "altercation_or_aggression": (
        "Visible physical aggression, fighting, threatening body language, "
        "or a struggle between two or more people."
    ),
    "employee_unauthorized_giveaway": (
        "An employee appears to hand merchandise to a person without any "
        "visible scan, payment, or transaction step at a register."
    ),
    "employee_policy_violation": (
        "An employee appears to be doing something against typical retail "
        "policy unrelated to giveaways - e.g. away from post for an "
        "extended period in a sensitive area, mishandling cash visibly, "
        "or similar - based only on what is visibly happening."
    ),
    "nothing_notable": (
        "Normal shopping or working behavior. No concerning activity."
    ),
}
 
# Confidence floor below which we don't bother surfacing the event to
# a human reviewer at all (still logged, just not "alerted")
ALERT_CONFIDENCE_FLOOR = 0.5
 
 
# --- Storage ---
EVENTS_DIR = os.path.join(PROJECT_ROOT, "events")
CLIPS_DIR = os.path.join(EVENTS_DIR, "clips")
EVENTS_LOG = os.path.join(EVENTS_DIR, "events.jsonl")

# Drop store footage here while pipeline.watcher is running
FOOTAGE_DIR = os.path.join(PROJECT_ROOT, "footage")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
