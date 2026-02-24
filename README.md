# Tiny Object Detector

A simple lightweight object detector using PyTorch and Faster R-CNN (MobileNetV3 backbone) with a Tkinter GUI.

---

## Features

- Detects **Books**, **Cups**, and **Bottles**.
- Filters detections by confidence threshold.
- Non-Maximum Suppression (NMS) and optional box merging.
- Visualizes detections in Tkinter with **distinct colors per class**.
- Easy-to-use GUI for opening images and running detection.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/coderanger428/ai-object-detector-simple.git
cd ai-object-detector-simple
```

2. Create environment and install dependencies:

```bash
conda create -n ai-global python=3.10
conda activate ai-global
pip install torch torchvision pillow
```

---

## Usage

Run the GUI:

```bash
python src/ui_tk.py
```

- Click Open Image to load an image.

- Check/uncheck classes to detect.

- Click Detect to see bounding boxes with class labels and scores.

---

## Configuration

- MODEL_PATH in ui_tk.py points to the trained model.

- DEVICE auto-detects GPU if available.

- Confidence threshold and NMS/merge settings can be modified in detect() function.

## Notes

- Non-Maximum Suppression reduces overlapping boxes.

- Optional box merging handles handles, overlapping object parts, or close objects.

- Box colors are mapped per class in draw_boxes().

## License

MIT License.