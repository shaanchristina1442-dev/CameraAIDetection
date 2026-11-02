"""
Review dashboard: shows events from events/events.jsonl with their frames
and Claude's judgment, and lets a human mark each one reviewed.

Usage:
    python -m dashboard.app      # then open http://127.0.0.1:5000
"""
import json
import os
import threading

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from pipeline import config

REVIEW_STATES = {"pending", "confirmed", "dismissed"}
# reviews live in their own file so the dashboard never rewrites the
# pipeline's append-only events.jsonl while it may be writing to it
REVIEWS_FILE = os.path.join(config.EVENTS_DIR, "reviews.json")

app = Flask(__name__)
reviews_lock = threading.Lock()


def read_events() -> list[dict]:
    if not os.path.exists(config.EVENTS_LOG):
        return []
    events = []
    with open(config.EVENTS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a half-written line while the pipeline is appending
    return events


def read_reviews() -> dict:
    try:
        with open(REVIEWS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/events')
def api_events():
    reviews = read_reviews()
    events = read_events()
    for e in events:
        e['review'] = reviews.get(e.get('event_id'), e.get('review', 'pending'))
    events.reverse()  # newest first
    return jsonify(events)


@app.route('/api/events/<event_id>/review', methods=['POST'])
def api_review(event_id):
    status = (request.get_json(silent=True) or {}).get('status')
    if status not in REVIEW_STATES:
        abort(400, f"status must be one of {sorted(REVIEW_STATES)}")

    if not any(e.get('event_id') == event_id for e in read_events()):
        abort(404)

    with reviews_lock:
        reviews = read_reviews()
        reviews[event_id] = status
        tmp = REVIEWS_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(reviews, f, indent=2)
        os.replace(tmp, REVIEWS_FILE)
    return jsonify({'event_id': event_id, 'review': status})


@app.route('/frames/<path:filename>')
def frames(filename):
    return send_from_directory(config.EVENTS_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True)
