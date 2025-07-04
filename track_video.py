from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video_path_in = 'anti_shoplift_system/shoplifting/shoplifting-3.mp4'
video_path_out = 'tracked_output.mp4'

cap = cv2.VideoCapture(video_path_in)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path_out, fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, classes=0)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Tracking complete! Video saved to {video_path_out}")