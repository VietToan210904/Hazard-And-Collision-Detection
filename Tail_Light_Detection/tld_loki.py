import os
import json
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO

# We are going to process the TLD-LOKI dataset for tail light detection
COCO_ROOT = "C:/Users/tonyh_yxuq8za/Desktop/tail_2/TLD-LOKI"
# we are going to save the processed images in this folder
OUTPUT_DIR = "C:/Users/tonyh_yxuq8za/Desktop/TONY_AI/Tail_Light_Detection/COCO"

# We are going to map the categories to the labels we want
CATEGORY_MAPPING = {
    "go": "brake_off",
    "car_brakeoff": "brake_off",
    "off": "brake_off",
    "brake": "brake_on",
    "car_brakeon": "brake_on",
    "left": "indicator_left",
    "right": "indicator_right",
}

# We are going to ensure the output directory exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# We are going to process the TLD-LOKI dataset
def process_tld_loki():
    print(" Processing the folder TLD-LOKI...")
    total = 0
    name_set = set()

    for group in os.listdir(COCO_ROOT):
        group_path = os.path.join(COCO_ROOT, group)
        if not os.path.isdir(group_path):
            continue

        for scenario in os.listdir(group_path):
            scenario_path = os.path.join(group_path, scenario)
            if not os.path.isdir(scenario_path):
                continue

            ann_path = None
            for f in os.listdir(scenario_path):
                if f.endswith(".coco.json"):
                    ann_path = os.path.join(scenario_path, f)
                    break
            if not ann_path or not os.path.exists(ann_path):
                continue

            try:
                coco = COCO(ann_path)
            except:
                print(f" Error when processing annotation: {ann_path}")
                continue

            cat_id_to_name = {
                cat["id"]: CATEGORY_MAPPING.get(cat["name"].lower(), "unknown")
                for cat in coco.loadCats(coco.getCatIds())
            }
            img_id_to_name = {img["id"]: img["file_name"] for img in coco.dataset["images"]}

            for ann in tqdm(coco.dataset["annotations"], desc=f"{scenario}"):
                cat_id = ann["category_id"]
                label = cat_id_to_name.get(cat_id, "unknown")

                img_id = ann["image_id"]
                filename = img_id_to_name.get(img_id)
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

                base_name = f"{img_id}_{ann['id']}.jpg"
                while base_name in name_set:
                    base_name = f"{img_id}_{ann['id']}_{total}.jpg"
                name_set.add(base_name)

                out_path = os.path.join(out_dir, base_name)
                cv2.imwrite(out_path, tail)
                total += 1

    print(f"\n DONE TLD-LOKI. Total images: {total}")

if __name__ == "__main__":
    process_tld_loki()
