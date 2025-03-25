from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import pickle
import io
import logging
import os
import base64

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
    # Create a more useful test image instead of just a black frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a grid pattern to make it more visible
    for i in range(0, 480, 30):
        cv2.line(dummy_frame, (0, i), (640, i), (0, 255, 0), 1)
    for i in range(0, 640, 30):
        cv2.line(dummy_frame, (i, 0), (i, 480), (0, 255, 0), 1)
    
    # Add text explaining the situation
    cv2.putText(dummy_frame, "Camera access not available on cloud server", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dummy_frame, "Please use the 'Upload Image' feature instead", (70, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


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

@app.post("/detect_from_image")
async def detect_from_image(file: UploadFile = File(...)):
    """Detect hand signs from an uploaded image."""
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []
            
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                    
                # Only proceed if we have landmarks
                if x_ and y_:
                    min_x, min_y = min(x_), min(y_)
                    
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)
                        
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                
                # Draw the prediction on the image
                cv2.putText(frame, f"Predicted: {predicted_character}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Return both the prediction and the annotated image
                _, img_encoded = cv2.imencode('.jpg', frame)
                return {
                    "alphabet": predicted_character,
                    "image": f"data:image/jpeg;base64,{base64.b64encode(img_encoded).decode('utf-8')}"
                }
            
        return {"alphabet": "", "message": "No hand detected in the image"}
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return JSONResponse(
            content={"error": "Failed to process image", "details": str(e)},
            status_code=500
        )
