# prepare_data.py
import json
import os
import shutil
from PIL import Image
import xml.etree.ElementTree as ET

# ===== CONFIG =====
COCO_JSON = "data/annotations/instances_val2017.json"
COCO_IMG_DIR = "data/val2017"
OUTPUT_IMG_DIR = "data/images"
OUTPUT_ANN_DIR = "data/annotations"

# Classes to keep
COCO_CLASS_IDS = {44: "bottle", 47: "cup", 84: "book"}

# ===== HELPER =====
def create_voc_xml(filename, width, height, objects):
    annotation = ET.Element("annotation")
    
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for obj in objects:
        o = ET.SubElement(annotation, "object")
        ET.SubElement(o, "name").text = obj["name"]
        b = ET.SubElement(o, "bndbox")
        ET.SubElement(b, "xmin").text = str(obj["xmin"])
        ET.SubElement(b, "ymin").text = str(obj["ymin"])
        ET.SubElement(b, "xmax").text = str(obj["xmax"])
        ET.SubElement(b, "ymax").text = str(obj["ymax"])

    return ET.ElementTree(annotation)

# ===== MAIN =====
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANN_DIR, exist_ok=True)

with open(COCO_JSON, "r") as f:
    coco = json.load(f)

# Map image_id -> file_name
images_dict = {img["id"]: img["file_name"] for img in coco["images"]}

# Map image_id -> list of objects
img_objects = {}
for ann in coco["annotations"]:
    cid = ann["category_id"]
    if cid not in COCO_CLASS_IDS:
        continue
    img_id = ann["image_id"]
    bbox = ann["bbox"]  # [x, y, width, height] in pixels
    x, y, w, h = bbox
    obj = {
        "name": COCO_CLASS_IDS[cid],
        "xmin": int(x),
        "ymin": int(y),
        "xmax": int(x + w),
        "ymax": int(y + h),
    }
    img_objects.setdefault(img_id, []).append(obj)

print(f"Found {len(img_objects)} images with target classes.")

# Copy images and write XML
for img_id, objects in img_objects.items():
    file_name = images_dict[img_id]
    src_path = os.path.join(COCO_IMG_DIR, file_name)
    dst_path = os.path.join(OUTPUT_IMG_DIR, file_name)
    shutil.copyfile(src_path, dst_path)

    img_path = dst_path
    with Image.open(img_path) as im:
        width, height = im.size

    xml_tree = create_voc_xml(file_name, width, height, objects)
    xml_tree.write(os.path.join(OUTPUT_ANN_DIR, file_name.replace(".jpg", ".xml")))

print("Dataset preparation complete!")