import os
import numpy as np
import cv2
import dlib
from flask import Flask, Response, render_template
from tensorflow.keras.models import load_model
from collections import deque
import time
from imutils import face_utils
import threading
import skfuzzy as fuzz
import skfuzzy.control as ctrl

app = Flask(__name__)

eye_model = load_model('eye_model.h5', compile=False)
yawn_model = load_model('yawn_model.h5', compile=False)

base_dir = os.path.dirname(os.path.abspath(__file__))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(base_dir, 'shape_predictor_68_face_landmarks.dat'))

eye_states = deque(maxlen=8)
yawn_states = deque(maxlen=8)
alert_message = "Normal"  
eye_closed_duration = 0  
parked = False  

eye_state = ctrl.Antecedent(np.arange(0, 2, 1), 'eye_state')
yawn_state = ctrl.Antecedent(np.arange(0, 2, 1), 'yawn_state')
alert_level = ctrl.Consequent(np.arange(0, 3, 1), 'alert_level')

eye_state['Open'] = fuzz.trimf(eye_state.universe, [0, 0, 1])
eye_state['Closed'] = fuzz.trimf(eye_state.universe, [1, 1, 2])
yawn_state['Not Yawning'] = fuzz.trimf(yawn_state.universe, [0, 0, 1])
yawn_state['Yawning'] = fuzz.trimf(yawn_state.universe, [1, 1, 2])
alert_level['Normal'] = fuzz.trimf(alert_level.universe, [0, 0, 1])
alert_level['Caution'] = fuzz.trimf(alert_level.universe, [1, 1, 2])
alert_level['Park'] = fuzz.trimf(alert_level.universe, [2, 2, 2])

rule1 = ctrl.Rule(eye_state['Open'] & yawn_state['Not Yawning'], alert_level['Normal'])
rule2 = ctrl.Rule(eye_state['Closed'] & yawn_state['Yawning'], alert_level['Caution'])
rule3 = ctrl.Rule(eye_state['Closed'] & yawn_state['Not Yawning'], alert_level['Park'])

alert_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
alert_sim = ctrl.ControlSystemSimulation(alert_ctrl)

class VideoCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.frame = None

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def stop(self):
        self.running = False
        self.cap.release()

video_thread = VideoCaptureThread()
video_thread.start()

def predict_image_state(image, model, labels):
    try:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (150, 150))
        img = np.array(img) / 255.0
        img = img.reshape(-1, 150, 150, 3)
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        return labels[class_idx]
    except Exception as e:
        print(f"Error predicting image state: {e}")
        return 'Unknown'

def get_alert_message(eye_state_value, yawn_state_value):
    global parked
    alert_sim.input['eye_state'] = eye_state_value
    alert_sim.input['yawn_state'] = yawn_state_value
    alert_sim.compute()
    return int(alert_sim.output['alert_level'])

def process_frame(frame):
    global eye_closed_duration
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)
    
    eye_labels = ['Closed', 'Open']
    yawn_labels = ['No yawn', 'Yawn']
    
    eye_state_value = 0
    yawn_state_value = 0

    for rect in faces:
        shape = predictor(gray_frame, rect)
        shape = face_utils.shape_to_np(shape)

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

        leftEye = shape[lStart:lEnd]
        mouth = shape[mStart:mEnd]

        leftEyeHull = cv2.convexHull(leftEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)

        left_eye_region = gray_frame[min(leftEye[:, 1]):max(leftEye[:, 1]), min(leftEye[:, 0]):max(leftEye[:, 0])]
        mouth_region = gray_frame[min(mouth[:, 1]):max(mouth[:, 1]), min(mouth[:, 0]):max(mouth[:, 0])]

        eye_state = predict_image_state(left_eye_region, eye_model, eye_labels)
        eye_state_value = 1 if eye_state == 'Closed' else 0
        yawn_state = predict_image_state(mouth_region, yawn_model, yawn_labels)
        yawn_state_value = 1 if yawn_state == 'Yawn' else 0

        cv2.putText(frame, f'Eye State: {eye_state}', (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Yawn State: {yawn_state}', (rect.left(), rect.bottom() + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    smoothed_eye_state_value = int(round(eye_state_value))
    smoothed_yawn_state_value = int(round(yawn_state_value))
    
    if smoothed_eye_state_value == 1:
        eye_closed_duration += 1
    else:
        eye_closed_duration = 0

    alert_level = get_alert_message(smoothed_eye_state_value, smoothed_yawn_state_value)
    alert_message = "Normal" if alert_level == 0 else "Caution" if alert_level == 1 else "Park"

    cv2.putText(frame, f'Alert: {alert_message}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def generate_frames():
    while True:
        if video_thread.frame is not None:
            frame = cv2.resize(video_thread.frame, (640, 480))
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        video_thread.stop()
