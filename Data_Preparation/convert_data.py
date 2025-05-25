import os
import json
import cv2
import shutil

# We are setting the paths for the dataset and output directories
DATASET_ROOT = "C:/Users/tonyh_yxuq8za/Desktop/tail_2"
YOLO_ROOT = "C:/Users/tonyh_yxuq8za/Desktop/TONY_AI/my_dataset"
IMG_OUT = os.path.join(YOLO_ROOT, "images", "train")
LBL_OUT = os.path.join(YOLO_ROOT, "labels", "train")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

counter = 0

# This function converts LabelMe JSON annotations to YOLO format
def convert_to_yolo(json_path, img_path):
    global counter
    print(f" Reading: {os.path.basename(json_path)}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f" Failed to load {json_path}: {e}")
        return

    image = cv2.imread(img_path)
    if image is None:
        print(f" Can not read image: {img_path}")
        return

    # We are getting the image dimensions
    H = data.get("imageHeight", image.shape[0])
    W = data.get("imageWidth", image.shape[1])

    yolo_lines = []

    for shape in data.get("shapes", []):
        try:
            points = shape["points"]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            x_center = ((x_min + x_max) / 2) / W
            y_center = ((y_min + y_max) / 2) / H
            width = (x_max - x_min) / W
            height = (y_max - y_min) / H

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        except Exception as e:
            print(f" Skipped a shape due to error: {e}")
            continue

    if not yolo_lines:
        print(f" Skipped (no valid annotations): {json_path}")
        return

    # This is to save the image and label in YOLO format
    base = f"{counter:06}"
    shutil.copy2(img_path, os.path.join(IMG_OUT, base + ".jpg"))
    with open(os.path.join(LBL_OUT, base + ".txt"), "w") as f:
        f.write("\n".join(yolo_lines))

    counter += 1
    print(f"[{counter:06}] Processed {os.path.basename(img_path)}")


# this is to walk through the dataset directory and process each JSON file
for dirpath, _, filenames in os.walk(DATASET_ROOT):
    for file in filenames:
        if file.endswith(".json"):
            json_path = os.path.join(dirpath, file)
            img_path = json_path.replace(".json", ".jpg")

            if os.path.exists(img_path):
                convert_to_yolo(json_path, img_path)
            else:
                print(f" Missing image for {file}")
