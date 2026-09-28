import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from pipeline.judge import Judge
from pipeline.trigger import TriggerEngine


app = Flask(__name__)
judge = Judge()
trigger = TriggerEngine()

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = trigger.get_frame()
            trigger = judge.analyze(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/analyze')
def analyze():
    def generate_spotter():
        while True:
            frame = trigger.get_feed()
            trigger = judge.analyze(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_spotter(), mimetype='multipart/x-mixed-replace; boundary=frame')



