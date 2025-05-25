import os
import json
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO

# We are going to process the TLD-LOKI dataset for tail light detection
LABELME_ROOT = "C:/Users/tonyh_yxuq8za/Desktop/tail_2"
COCO_ROOT = "C:/Users/tonyh_yxuq8za/Desktop/tail_2/TLD-LOKI"
OUTPUT_DIR = "C:/Users/tonyh_yxuq8za/Desktop/TONY_AI/Tail_Light_Detection/Data"

# We are going to map the categories to the labels we want
CATEGORY_NAME_MAP = {
    "brake": "brake_on",
    "left": "indicator_left",
    "right": "indicator_right"
}

# We are going to ensure the output directory exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# We are going to process the dataset with LabelMe formats
def process_labelme():
    print("\n LabelMe...")
    total = 0
    idx = 0
    for root, _, files in os.walk(LABELME_ROOT):
        if "TLD-LOKI" in root:
            continue
        folder_name = os.path.basename(root)
        for file in files:
            if file.endswith(".json") and "annotations.coco" not in file:
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                except:
                    print(f" Error Reading File: {json_path}")
                    continue

                img_path = os.path.join(root, data.get("imagePath", ""))
                if not os.path.exists(img_path):
                    print(f" No Images Found: {img_path}")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                for shape in data.get("shapes", []):
                    if not shape["label"].startswith("car_Brake"):
                        continue

                    brake = shape["label"].split("_")[1].lower()
                    turn = shape.get("attributes", {}).get("turn_signal", "off").lower()

                    pts = shape["points"]
                    x1, x2 = int(min(p[0] for p in pts)), int(max(p[0] for p in pts))
                    y1, y2 = int(min(p[1] for p in pts)), int(max(p[1] for p in pts))

                    tail = img[y1:y2, x1:x2]
                    if tail.shape[0] == 0 or tail.shape[1] == 0:
                        continue

                    tail = cv2.resize(tail, (64, 64))

                    for folder in [f"brake_{brake}", f"indicator_{turn}"]:
                        out_dir = os.path.join(OUTPUT_DIR, folder)
                        ensure_dir(out_dir)
                        save_name = f"labelme_{folder_name}_{os.path.splitext(file)[0]}_{idx}.jpg"
                        cv2.imwrite(os.path.join(out_dir, save_name), tail)
                        print(f"[✓] {save_name}")
                        idx += 1
                        total += 1
    print(f" DONE LabelMe. Total Images: {total}")
    
# We are going to process the dataset with COCO formats
def process_coco():
    print("\n COCO...")
    total = 0
    for group in os.listdir(COCO_ROOT):
        group_path = os.path.join(COCO_ROOT, group)
        if not os.path.isdir(group_path):
            continue
        for scenario in os.listdir(group_path):
            scenario_path = os.path.join(group_path, scenario)
            ann_path = os.path.join(scenario_path, "annotations.coco.json")
            if not os.path.exists(ann_path):
                continue
            try:
                coco = COCO(ann_path)
            except:
                print(f" ERROR COCO file: {ann_path}")
                continue

            cat_map = {
                cat["id"]: CATEGORY_NAME_MAP[cat["name"]]
                for cat in coco.loadCats(coco.getCatIds())
                if cat["name"] in CATEGORY_NAME_MAP
            }
            img_id_map = {img["id"]: img["file_name"] for img in coco.dataset["images"]}
            print(f"Scenario: {scenario} — {len(coco.dataset['annotations'])} annotation(s)")

            for ann in coco.dataset["annotations"]:
                if ann["category_id"] not in cat_map:
                    continue
                label = cat_map[ann["category_id"]]
                img_id = ann["image_id"]
                filename = img_id_map.get(img_id, None)
                if not filename:
                    continue

                img_path = os.path.join(scenario_path, filename)
                if not os.path.exists(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                x, y, w, h = map(int, ann["bbox"])
                tail = img[y:y + h, x:x + w]
                if tail.shape[0] == 0 or tail.shape[1] == 0:
                    continue

                tail = cv2.resize(tail, (64, 64))
                out_dir = os.path.join(OUTPUT_DIR, label)
                ensure_dir(out_dir)
                save_name = f"coco_{scenario}_{img_id}_{ann['id']}.jpg"
                cv2.imwrite(os.path.join(out_dir, save_name), tail)
                print(f"{label} → {save_name}")
                total += 1

    print(f" DONE COCO. Total Images: {total}")
    
# We are going to run the processing functions
if __name__ == "__main__":
    process_labelme()
    process_coco()
    print("\n DONE ALL")
