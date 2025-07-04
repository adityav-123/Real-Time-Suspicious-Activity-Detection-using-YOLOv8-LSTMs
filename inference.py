import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

action_model = load_model('action_recognition_model.h5')
tracker_model = YOLO('yolov8n.pt')

with open('action_classes.txt', 'r') as f:
    action_classes = [line.strip() for line in f]

VIDEO_PATH_IN = 'anti_shoplift_system/normal_walk/normal-74.mp4' 
VIDEO_PATH_OUT = 'inference_output.mp4'

MAX_FRAMES = 20
IMG_SIZE = 128
track_history = {}

cap = cv2.VideoCapture(VIDEO_PATH_IN)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_PATH_OUT, fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results = tracker_model.track(frame, persist=True, classes=0)

    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    annotated_frame = results[0].plot()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box

        roi = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

        if roi.size == 0: continue
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi_normalized = roi_resized / 255.0

        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append(roi_normalized)

        if len(track_history[track_id]) > MAX_FRAMES:
            track_history[track_id].pop(0)

        if len(track_history[track_id]) == MAX_FRAMES:
            frames_to_predict = np.expand_dims(np.array(track_history[track_id]), axis=0)
            prediction = action_model.predict(frames_to_predict)
            predicted_class_index = np.argmax(prediction)
            predicted_action = action_classes[predicted_class_index]
            confidence = np.max(prediction) * 100

            text = f"{predicted_action} ({confidence:.0f}%)"
            cv2.putText(annotated_frame, text, (int(x - w / 2), int(y - h / 2) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(annotated_frame)

cap.release()
out.release()

print(f"✅ Inference complete! Video saved to {VIDEO_PATH_OUT}")