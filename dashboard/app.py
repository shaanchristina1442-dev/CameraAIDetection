"""
Review dashboard: shows judged events with their frames and Claude's
judgment, and lets a human mark each one reviewed.

Reads and writes only through storage.repository - no SQL here. Events come
back in the same shape the old JSONL records had, so the templates and the
front-end JavaScript did not change when the storage did.

Usage:
    python -m dashboard.app      # then open http://127.0.0.1:5000
"""
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from pipeline import config
from storage import repository

app = Flask(__name__)
repository.init()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/events')
def api_events():
    # already newest-first, with each event's latest review state applied
    return jsonify(repository.list_events())


@app.route('/api/stats')
def api_stats():
    return jsonify(repository.stats())


@app.route('/api/events/<event_id>/review', methods=['POST'])
def api_review(event_id):
    status = (request.get_json(silent=True) or {}).get('status')
    if status not in repository.REVIEW_STATES:
        abort(400, f"status must be one of {sorted(repository.REVIEW_STATES)}")
    if not repository.event_exists(event_id):
        abort(404)
    # appends a row; earlier decisions are kept, never overwritten
    return jsonify(repository.set_review(event_id, status))


@app.route('/api/events/<event_id>/history')
def api_history(event_id):
    if not repository.event_exists(event_id):
        abort(404)
    return jsonify(repository.review_history(event_id))


@app.route('/frames/<path:filename>')
def frames(filename):
    return send_from_directory(config.EVENTS_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True)
