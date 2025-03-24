from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import pickle
import io
import logging
import os

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hand-sign-detection-frontend.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {i: chr(65 + i) for i in range(26)}  # Map numbers to A-Z

# Global capture object
is_production = os.environ.get('RENDER', '') == 'true'

if is_production:
    # In production, we won't use a real webcam
    # Just create a dummy capture or use a static image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap = None
else:
    # In development, use the webcam
    cap = cv2.VideoCapture(0)

@app.on_event("shutdown")
def shutdown_event():
    logging.info("Releasing video capture resources.")
    if cap is not None:
        cap.release()

def gen_frames():
    """Generate video frames for streaming."""
    while True:
        if cap is not None:
            success, frame = cap.read()
            if not success:
                break
        else:
            # In production, use a dummy frame
            frame = dummy_frame.copy()
            success = True
            
        if success:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/video_feed")
def video_feed():
    """Endpoint to stream video feed."""
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detect")
def detect_hand_sign():
    """Detect hand signs from the webcam."""
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame.")
            return JSONResponse(content={"error": "Failed to capture frame."}, status_code=500)
    else:
        # In production, use a dummy frame or test image
        frame = dummy_frame.copy()
        ret = True

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_, y_ = [], []

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            min_x, min_y = min(x_), min(y_)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            return {"alphabet": predicted_character}

    return {"alphabet": ""}
