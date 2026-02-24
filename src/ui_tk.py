# src/ui_tk.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from dataset import get_transform, CLASS_TO_IDX
from torchvision.ops import nms

# ===== CONFIG =====
MODEL_PATH = "./models/model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3 + 1

# ===== LOAD MODEL =====
model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# ===== HELPER FUNCTIONS =====
def draw_boxes(image, boxes, labels, scores, selected_classes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    idx_to_class = {v:k for k,v in CLASS_TO_IDX.items()}

    for box, label, score in zip(boxes, labels, scores):
        cls_name = idx_to_class.get(label.item(), "").lower()
        if cls_name not in selected_classes:
            continue
        if score < 0.5:
            continue
        xmin, ymin, xmax, ymax = box.tolist()  # convert tensor to list
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin-10), f"{cls_name} ({score:.2f})", fill="red", font=font)
    return image

def open_image():
    global current_image, tk_image, displayed_image
    file_path = filedialog.askopenfilename(filetypes=[("JPEG Files","*.jpg"),("PNG Files","*.png")])
    if file_path:
        current_image = Image.open(file_path).convert("RGB")
        displayed_image = current_image.copy()
        tk_image = ImageTk.PhotoImage(displayed_image)
        canvas.config(width=tk_image.width(), height=tk_image.height())
        canvas.create_image(0,0,anchor="nw",image=tk_image)
        canvas.image = tk_image

def merge_boxes(boxes, labels, scores, axis_overlap_thresh=0.5, max_center_dist=50):
    """Merge overlapping boxes of the same class if overlap along an axis is high and centers are close."""
    merged_boxes, merged_labels, merged_scores = [], [], []
    used = torch.zeros(len(boxes), dtype=torch.bool)

    for i in range(len(boxes)):
        if used[i]:
            continue
        box_i = boxes[i].clone()
        score_i = scores[i].item()
        label_i = labels[i].item()

        for j in range(i+1, len(boxes)):
            if used[j] or labels[j].item() != label_i:
                continue

            # Compute axis overlaps
            x1 = max(box_i[0], boxes[j][0])
            x2 = min(box_i[2], boxes[j][2])
            x_overlap = max(0, x2 - x1) / min(box_i[2]-box_i[0], boxes[j][2]-boxes[j][0])

            y1 = max(box_i[1], boxes[j][1])
            y2 = min(box_i[3], boxes[j][3])
            y_overlap = max(0, y2 - y1) / min(box_i[3]-box_i[1], boxes[j][3]-boxes[j][1])

            # Compute center distance
            cx_i, cy_i = (box_i[0]+box_i[2])/2, (box_i[1]+box_i[3])/2
            cx_j, cy_j = (boxes[j][0]+boxes[j][2])/2, (boxes[j][1]+boxes[j][3])/2
            center_dist = ((cx_i - cx_j)**2 + (cy_i - cy_j)**2)**0.5

            if (x_overlap > axis_overlap_thresh or y_overlap > axis_overlap_thresh) and center_dist < max_center_dist:
                # merge
                box_i = torch.tensor([
                    min(box_i[0], boxes[j][0]),
                    min(box_i[1], boxes[j][1]),
                    max(box_i[2], boxes[j][2]),
                    max(box_i[3], boxes[j][3])
                ])
                score_i = max(score_i, scores[j].item())
                used[j] = True

        merged_boxes.append(box_i)
        merged_labels.append(label_i)
        merged_scores.append(score_i)
    if len(merged_boxes) == 0:
        return torch.empty((0,4)), torch.empty((0,), dtype=torch.int64), torch.empty((0,))

    return torch.stack(merged_boxes), torch.tensor(merged_labels), torch.tensor(merged_scores)

def detect():
    global tk_image, displayed_image
    if current_image is None:
        messagebox.showinfo("Info", "Please open an image first!")
        return

    # Transform
    transform = get_transform(train=False)
    img_tensor = transform(current_image).unsqueeze(0).to(DEVICE)

    # Model inference
    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]["boxes"].cpu()
    labels = outputs[0]["labels"].cpu()
    scores = outputs[0]["scores"].cpu()

    # Filter by confidence first
    conf_thresh = 0.6
    keep_conf = scores > conf_thresh
    boxes = boxes[keep_conf]
    labels = labels[keep_conf]
    scores = scores[keep_conf]

    # Apply NMS
    keep_nms = nms(boxes, scores, iou_threshold=0.6)
    boxes = boxes[keep_nms]
    labels = labels[keep_nms]
    scores = scores[keep_nms]

    boxes, labels, scores = merge_boxes(boxes, labels, scores, axis_overlap_thresh=0.6)
    print("Kept boxes after conf + NMS + merge:", boxes)
    print("Kept labels:", labels)
    print("Kept scores:", scores)

    selected_classes = [k.lower() for k,v in class_vars.items() if v.get()]

    # Draw boxes
    displayed_image = draw_boxes(current_image.copy(), boxes, labels, scores, selected_classes)

    tk_image = ImageTk.PhotoImage(displayed_image)
    canvas.delete("all")
    canvas.config(width=tk_image.width(), height=tk_image.height())
    canvas.create_image(0, 0, anchor="nw", image=tk_image)
    canvas.image = tk_image

# ===== TKINTER UI =====
root = tk.Tk()
root.title("Tiny Object Detector")

btn_open = tk.Button(root, text="Open Image", command=open_image)
btn_open.pack(pady=5)

btn_detect = tk.Button(root, text="Detect", command=detect)
btn_detect.pack(pady=5)

class_vars = {"Book": tk.IntVar(value=1), "Cup": tk.IntVar(value=1), "Bottle": tk.IntVar(value=1)}
for name, var in class_vars.items():
    tk.Checkbutton(root, text=name, variable=var).pack(anchor="w")

canvas = tk.Canvas(root, width=600, height=400)
canvas.pack()

current_image = None
displayed_image = None
tk_image = None

root.mainloop()