import cv2
import torch
import numpy as np
import time
class YOLOv5Detector:
    def __init__(self, model_path="yolov5s.pt", conf_threshold=0.25):
        print(f"Loading YOLOv5 model from: {model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=self.device)
        self.model.conf = conf_threshold
        self.model.iou = 0.45
        self.model.classes = None
        print("Model loaded successfully")
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        try:
            result_df = results.pandas().xyxy[0]
            for i in range(len(result_df)):
                row = result_df.iloc[i]
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                conf = float(row['confidence'])
                cls_id = int(row['class'])
                cls_name = row['name']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detections.append((x1, y1, x2, y2, conf, cls_id, cls_name))
        except Exception as e:
            print(f"Error processing results: {e}")
        return frame, detections
def process_video(video_path, model_path, output_path=None):
    detector = YOLOv5Detector(model_path, conf_threshold=0.25)
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {width}x{height} @ {fps}fps")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if output_path else None
    frame_count, processing_times = 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
        start_time = time.time()
        frame_with_detections, detections = detector.detect(frame)
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        fps_text = f"FPS: {1/processing_time:.1f}" if processing_time > 0 else "FPS: N/A"
        cv2.putText(frame_with_detections, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame_with_detections, f"Detections: {len(detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("YOLOv5 Detection", frame_with_detections)
        if out:
            out.write(frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
        if frame_count % 10 == 0:
            avg_time = sum(processing_times[-10:]) / min(10, len(processing_times[-10:]))
            fps_avg = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Processed {frame_count} frames. Avg time: {avg_time:.3f}s ({fps_avg:.1f} FPS)")
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Processed {frame_count} frames with avg time of {avg_time:.3f}s ({1/avg_time:.2f} FPS)")
if __name__ == "__main__":
    video_path, model_path, output_path = "people.mp4", "yolov5s.pt", "output.mp4"
    process_video(video_path, model_path, output_path)
