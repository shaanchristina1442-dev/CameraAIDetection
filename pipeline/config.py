"""
Central configuration for the store monitoring prototype.
 
IMPORTANT CONTEXT FOR ANYONE EXTENDING THIS:
This system is designed as a HUMAN-IN-THE-LOOP ALERTING TOOL, not an
autonomous accusation system. It surfaces moments worth a human's
attention. It does not label anyone a "thief" or take any automated
action. All outputs should be treated as "worth reviewing", not "confirmed".
"""
 
# --- Trigger layer (YOLO) settings ---
 
# How often to run object detection (every Nth frame). Lower = more
# sensitive / slower. For a demo on a short clip, 1-3 is fine.
FRAME_SKIP = 2
 
# Confidence threshold for YOLO person/object detections
YOLO_CONF_THRESHOLD = 0.45
 
# Classes from the COCO dataset (what the pretrained YOLO model knows)
# that are relevant triggers. "person" is the main one we care about;
# others help us reason about hand-near-object events.
RELEVANT_CLASSES = ["person", "backpack", "handbag", "suitcase", "bottle", "cell phone"]
 
# Motion energy threshold (0-1 normalized) above which we consider a
# region "active" and worth a closer look
MOTION_THRESHOLD = 0.02
 
# Minimum seconds between two triggered clips to avoid spamming the
# judgment layer with near-duplicate events
MIN_SECONDS_BETWEEN_EVENTS = 4
 
# How many seconds of context to grab around a trigger (before/after)
CLIP_PADDING_SECONDS = 2
 
# How many frames to sample from a flagged clip to send to the
# judgment layer (more frames = better judgment, more $ per event)
FRAMES_PER_JUDGMENT = 5
 
 
# --- Judgment layer (Claude vision) settings ---
 
CLAUDE_MODEL = "claude-sonnet-4-6"
 
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
EVENTS_DIR = "events"
CLIPS_DIR = "events/clips"
EVENTS_LOG = "events/events.jsonl"