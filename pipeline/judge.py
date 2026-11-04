"""
Judgment layer: takes a candidate event (a handful of frames flagged
by the cheap trigger layer) and asks Claude's vision capability to
describe what's happening and classify it.

This is intentionally framed as DESCRIPTION + CLASSIFICATION FOR HUMAN
REVIEW, not a verdict. The output schema includes a confidence score
and a plain-language rationale specifically so a human reviewer can
sanity-check the model's reasoning rather than just trusting a label.

Two things here are load-bearing and easy to break:

1. The response shape is enforced by the API (output_config.format), not
   by parsing prose. The category enum is generated from
   config.DETECTION_CATEGORIES, so the model cannot return a category
   this system does not know about, and the two can never drift apart.

2. max_tokens must leave room for thinking. Thinking is adaptive by
   default on current models and spends the same budget as the answer,
   so a budget sized for the JSON alone returns a truncated response and
   every event fails. That case is detected explicitly below rather than
   surfacing as a parse error.
"""
import base64
import json

import cv2
import anthropic

from . import config

_client = None


def _get_client():
    # created on first use so importing this module doesn't require an API key
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def crop_to_people(frame, boxes):
    """
    Crop to the region containing the detected people, with padding.

    A whole frame scaled to 640px wide leaves a distant person only a few
    dozen pixels tall, and the detail that decides the category - a hand
    moving to a pocket, an item leaving a shelf - smaller still. The trigger
    layer already located the people, so spend the resolution there.

    Returns the frame unchanged when cropping would not help: no people
    found, cropping disabled, or the people already fill most of the frame.

    `boxes` is a list of (x1, y1, x2, y2) pixel tuples in `frame`'s space.
    """
    if not config.CROP_TO_PEOPLE or not boxes:
        return frame

    h, w = frame.shape[:2]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    if x2 <= x1 or y2 <= y1:
        return frame

    # One crop covering everyone, not one per person: for an altercation the
    # space between two people IS the evidence. Revisit if busy shots dilute
    # single-person events.
    pad_x = int((x2 - x1) * config.CROP_PADDING)
    pad_y = int((y2 - y1) * config.CROP_PADDING)
    x1, x2 = x1 - pad_x, x2 + pad_x
    y1, y2 = y1 - pad_y, y2 + pad_y

    # keep a floor on the crop so a reviewer still sees surroundings
    min_w, min_h = int(w * config.CROP_MIN_FRACTION), int(h * config.CROP_MIN_FRACTION)
    if x2 - x1 < min_w:
        cx = (x1 + x2) // 2
        x1, x2 = cx - min_w // 2, cx - min_w // 2 + min_w
    if y2 - y1 < min_h:
        cy = (y1 + y2) // 2
        y1, y2 = cy - min_h // 2, cy - min_h // 2 + min_h

    # slide back inside the frame rather than clipping, so the crop keeps its size
    if x1 < 0:
        x2, x1 = x2 - x1, 0
    if y1 < 0:
        y2, y1 = y2 - y1, 0
    if x2 > w:
        x1, x2 = x1 - (x2 - w), w
    if y2 > h:
        y1, y2 = y1 - (y2 - h), h
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if (x2 - x1) * (y2 - y1) >= config.CROP_SKIP_ABOVE * w * h:
        return frame

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else frame


def _frame_to_base64(frame, max_width=640) -> str:
    h, w = frame.shape[:2]

    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")

    return base64.b64encode(buf).decode('utf-8')


def _response_schema() -> dict:
    """Built from config so the categories can never drift out of sync."""
    return {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(config.DETECTION_CATEGORIES.keys()),
            },
            "confidence": {"type": "number"},
            "visible_description": {"type": "string"},
            "reasoning": {"type": "string"},
            "recommend_human_review": {"type": "boolean"},
        },
        "required": [
            "category", "confidence", "visible_description",
            "reasoning", "recommend_human_review",
        ],
        "additionalProperties": False,
    }


def _build_prompt(frames, context) -> str:
    category_lines = "\n".join(f'- "{key}": {desc}' for key, desc in config.DETECTION_CATEGORIES.items())
    cropped = " They are cropped around the people the motion detector found, so the framing is tighter than the camera's full view." if config.CROP_TO_PEOPLE else ""

    return f"""You are assisting a retail loss-prevention team by describing
what is visibly happening in a short sequence of security camera frames.
The frames are sampled in time order from a single flagged moment.{cropped}

Your job is description and classification for HUMAN REVIEW ONLY. You are
not making an accusation, issuing a verdict, or identifying any individual.
Do not guess at intent or identity. Describe only visible, observable actions.
If the frames are ambiguous, low quality, or don't show enough to tell,
say so plainly and lower your confidence accordingly.

Why the motion detector flagged this moment: {context}

Classify the scene into exactly one of these categories:
{category_lines}

Set "confidence" between 0.0 and 1.0. Reserve high confidence for frames
where the action is unambiguous. Set "recommend_human_review" to true when
a person should look at this, including when the frames are too unclear to
judge confidently.
"""


def _failure(code: str, description: str, reasoning: str, review: bool = True, **extra) -> dict:
    out = {
        "category": "nothing_notable",
        "confidence": 0.0,
        "visible_description": description,
        "reasoning": reasoning,
        "recommend_human_review": review,
        "error": code,
    }
    out.update(extra)
    return out


def judge_event(frames, context):
    """
    frames: list of OpenCV BGR numpy arrays sampled from one trigger event,
            already cropped by crop_to_people if cropping is enabled.
    Returns the parsed classification dict, or an error dict on failure.
    """
    if not frames:
        return _failure("no_frames", "No frames available.",
                        "No frames were extracted for this event.", review=False)

    content = [{"type": "text", "text": _build_prompt(frames, context)}]

    for i, frame in enumerate(frames):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _frame_to_base64(frame),
            },
        })
        content.append({"type": "text", "text": f"(frame {i + 1} of {len(frames)}, in time order)"})

    try:
        response = _get_client().messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.JUDGE_MAX_TOKENS,
            output_config={
                "effort": config.JUDGE_EFFORT,
                "format": {"type": "json_schema", "schema": _response_schema()},
            },
            messages=[{"role": "user", "content": content}],
        )

        # Check why generation stopped before reading content. A refusal and a
        # truncation both arrive as HTTP 200 with unusable content, and without
        # this they would surface as confusing JSON parse errors.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            return _failure("refusal", "The model declined to analyse these frames.",
                            f"Safety classifier declined this request (category: {category}).")

        if response.stop_reason == "max_tokens":
            return _failure(
                "response_truncated",
                "The model's response was cut off.",
                f"Hit the {config.JUDGE_MAX_TOKENS}-token limit before finishing. "
                "Raise JUDGE_MAX_TOKENS, or lower JUDGE_EFFORT so less of the budget goes to thinking.")

        # output_config.format guarantees the text block holds schema-valid JSON
        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            return _failure("empty_response", "The model returned no text.",
                            f"Response contained no text block (stop_reason: {response.stop_reason}).")

        parsed = json.loads(text)
        try:
            parsed["confidence"] = min(1.0, max(0.0, float(parsed.get("confidence", 0.0))))
        except (TypeError, ValueError):
            parsed["confidence"] = 0.0
        return parsed

    except json.JSONDecodeError as e:
        return _failure("json_parse_failed", "Model response could not be parsed.", str(e))
    except anthropic.APIError as e:
        return _failure("api_error", "API call failed.", str(e))
