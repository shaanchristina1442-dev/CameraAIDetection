"""
Judgment layer: takes a candidate event (a handful of frames flagged
by the cheap trigger layer) and asks Claude's vision capability to
describe what's happening and classify it.

This is intentionally framed as DESCRIPTION + CLASSIFICATION FOR HUMAN
REVIEW, not a verdict. The output schema includes a confidence score
and a plain-language rationale specifically so a human reviewer can
sanity-check the model's reasoning rather than just trusting a label.
"""
import base64
import json

import cv2
import anthropic

from . import config

client = anthropic.Anthropic()


def _frame_to_base64(frame, max_width=640) -> str:
    h, w = frame.shape[:2]

    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")

    return base64.b64encode(buf).decode('utf-8')


def _build_prompt(frames, context) -> str:
    category_lines = "\n".join(f'- "{key}": {desc}' for key, desc in config.DETECTION_CATEGORIES.items())

    return f"""You are assisting a retail loss-prevention team by describing
what is visibly happening in a short sequence of security camera frames.
The frames are sampled in time order from a single flagged moment.

Your job is description and classification for HUMAN REVIEW ONLY. You are
not making an accusation, issuing a verdict, or identifying any individual.
Do not guess at intent or identity. Describe only visible, observable actions.
If the frames are ambiguous, low quality, or don't show enough to tell,
say so plainly and lower your confidence accordingly.

Classify the scene into exactly one of these categories:
{category_lines}

Respond with ONLY a JSON object (no markdown fences, no preamble) in this
exact shape:
{{
  "category": "<one of the category keys above>",
  "confidence": <float 0.0-1.0>,
  "visible_description": "<2-3 sentences describing only what is visibly happening, neutrally>",
  "reasoning": "<1-2 sentences on why you chose this category over others>",
  "recommend_human_review": <true/false>
}}
"""


def judge_event(frames, context):
    """
    frames: list of OpenCV BGR numpy arrays sampled from one trigger event.
    Returns the parsed classification dict, or an error dict on failure.
    """
    if not frames:
        return {
            "category": "nothing_notable",
            "confidence": 0.0,
            "visible_description": "No frames available.",
            "reasoning": "No frames were extracted for this event.",
            "recommend_human_review": False,
            "error": "no_frames",
        }

    content = [{"type": "text", "text": _build_prompt(frames, context)}]

    for i, frame in enumerate(frames):
        b64 = _frame_to_base64(frame)

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })
        content.append({"type": "text", "text": f"(frame {i + 1} of {len(frames)}, in time order)"})

    raw_text = None
    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=450,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        if raw_text.startswith("`"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        if parsed.get("category") not in config.DETECTION_CATEGORIES:
            parsed["_warning"] = f"unrecognized category returned: {parsed.get('category')}"

        return parsed

    except json.JSONDecodeError as e:
        return {
            "category": "nothing_notable",
            "confidence": 0.0,
            "visible_description": "Model response could not be parsed.",
            "reasoning": str(e),
            "recommend_human_review": True,
            "error": "json_parse_failed",
            "raw_response": raw_text,
        }
    except anthropic.APIError as e:
        return {
            "category": "nothing_notable",
            "confidence": 0.0,
            "visible_description": "API call failed.",
            "reasoning": str(e),
            "recommend_human_review": True,
            "error": "api_error",
        }
