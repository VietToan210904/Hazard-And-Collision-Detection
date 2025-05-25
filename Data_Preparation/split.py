import os
import random
import shutil

# This is the base path for the YOLO dataset
YOLO_ROOT = "C:/Users/tonyh_yxuq8za/Desktop/TONY_AI/my_dataset"

# This is the path for raw images and labels
IMAGE_RAW = os.path.join(YOLO_ROOT, "images", "raw")
LABEL_RAW = os.path.join(YOLO_ROOT, "labels", "raw")

# This is the output path for images and labels
OUT_IMG = os.path.join(YOLO_ROOT, "images")
OUT_LBL = os.path.join(YOLO_ROOT, "labels")
SPLITS = ["train", "val", "test"]

# This is the ratio for splitting the dataset
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# We are splitting the dataset into train, validation, and test sets by using the following function
def split_dataset():
    print(" Looking for raw images in:", IMAGE_RAW)
    image_files = [f for f in os.listdir(IMAGE_RAW) if f.endswith(".jpg")]
    total_images = len(image_files)

    if total_images == 0:
        print(" No images found in the raw images folder.")
        return

    print(f" Found {total_images} images. Splitting...")

    random.shuffle(image_files)
    n_train = int(total_images * train_ratio)
    n_val = int(total_images * val_ratio)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:]
    }

    # We are ceating the output directories
    for split in SPLITS:
        os.makedirs(os.path.join(OUT_IMG, split), exist_ok=True)
        os.makedirs(os.path.join(OUT_LBL, split), exist_ok=True)

    # we are copying the images and labels to the output directories
    for split, files in splits.items():
        print(f" Copying {split} set ({len(files)} images)...")
        for img_file in files:
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"

            img_src = os.path.join(IMAGE_RAW, img_file)
            lbl_src = os.path.join(LABEL_RAW, lbl_file)
            img_dst = os.path.join(OUT_IMG, split, img_file)
            lbl_dst = os.path.join(OUT_LBL, split, lbl_file)

            try:
                shutil.copy(img_src, img_dst)
                shutil.copy(lbl_src, lbl_dst)
            except Exception as e:
                print(f" Error copying {img_file}: {e}")

        print(f" {split}: {len(files)} files copied.")

    print("Dataset split completed!")

# This is the main function to run the script
if __name__ == "__main__":
    print("Starting dataset split...")
    split_dataset()
