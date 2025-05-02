# YOLOv5 Video Object Detection

This project uses YOLOv5 (via PyTorch Hub) to perform real-time object detection on video files. Detected objects are highlighted with bounding boxes and class labels, and the processed video can be saved to disk.

## Features

- Loads a YOLOv5 model (default: `yolov5s.pt`)
- Processes a video file frame by frame
- Draws bounding boxes and class labels on detected objects
- Displays FPS and detection count on each frame
- Optionally saves the output video with detections

## Requirements

- Python 3.7+
- PyTorch
- OpenCV (`opencv-python`)
- NumPy

## Installation

1. **Clone the repository** (if applicable) or copy the script to your working directory.

2. **Install dependencies:**

   ```bash
   pip install torch torchvision opencv-python numpy
   ```

3. **Download YOLOv5 model weights:**

   The script will automatically download the `yolov5s.pt` weights if not present. You can also download other YOLOv5 weights from the [Ultralytics YOLOv5 releases](https://github.com/ultralytics/yolov5/releases).

## Usage

1. **Prepare your video file** (e.g., `traffic.mp4`) and place it in the same directory as the script, or provide the full path.

2. **Run the script:**

   ```bash
   python your_script_name.py
   ```

   By default, the script uses:
   - `traffic.mp4` as the input video
   - `yolov5s.pt` as the model weights
   - `output.mp4` as the output video

   You can modify these values in the `if __name__ == "__main__":` block.

3. **Controls:**
   - The video window will display detections in real time.
   - Press `q` to quit early.

## Customization

- **Change the model:**  
  Replace `"yolov5s.pt"` with another YOLOv5 model path (e.g., `"yolov5m.pt"` or a custom-trained model).
- **Change the video:**  
  Replace `"traffic.mp4"` with your own video file.
- **Change the output path:**  
  Set `output_path` to `None` if you do not want to save the output video.

## Notes

- The script will use GPU (CUDA) if available, otherwise it will fall back to CPU.
- The first time you run the script, PyTorch Hub will download the YOLOv5 repository and model weights if not already present.
- The script displays FPS and the number of detections per frame.

## Troubleshooting

- If you encounter issues with PyTorch Hub or model loading, ensure you have a stable internet connection for the first run.
- For custom models, ensure your `.pt` file is compatible with the Ultralytics YOLOv5 format.

## License

This project uses YOLOv5 by Ultralytics, which is licensed under the GNU Affero General Public License v3.0.

---

**Enjoy detecting objects in your videos!**

If you need further customization or encounter any issues, feel free to ask.
