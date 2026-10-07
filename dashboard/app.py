import os
import cv2
from flask import Flask, render_template, Response, jsonify
from pipeline import config
from pipeline.judge import judge_event
from pipeline.trigger import TriggerEngine

VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'sample_clips', 'synthetic_demo.mp4')

app = Flask(__name__)
engine = TriggerEngine()


def sample_frames(frames, n):
    """Pick n frames spread evenly across the event."""
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(VIDEO_PATH)
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                continue
            ok, jpg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analyze')
def analyze():
    events = engine.scan_video(VIDEO_PATH)
    results = []
    for e in events:
        frames = sample_frames(e.sampled_frames, config.FRAMES_PER_JUDGMENT)
        results.append({
            'start_sec': e.start_time_sec,
            'end_sec': e.end_time_sec,
            'reason': e.reason,
            'judgment': judge_event(frames, context=e.reason),
        })
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
