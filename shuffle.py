import os
import random
import shutil
from glob import glob

def shuffle_dataset(dataset_path, split_ratio=0.6):
    """
    Shuffles and splits dataset into train/test (keeps valid completely unchanged—no scan or move).
    Args:
        dataset_path (str): Root path, e.g., './uploads/dataset'
        split_ratio (float): Fraction for train (rest to test).
    Returns:
        tuple: (train_count, test_count)
    """
    print(f"🔀 Shuffling dataset at {dataset_path}...")
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    
    # FIXED: Collect ONLY from train/test (exclude valid entirely to keep it same)
    for sub in ["train", "test"]:  # Skip "valid"
        sub_dir = os.path.join(dataset_path, sub, "images")
        if os.path.exists(sub_dir):
            for ext in image_extensions:
                images.extend(glob(os.path.join(sub_dir, ext)))
    
    print(f"Total non-valid images for shuffle: {len(images)}")
    if len(images) == 0:
        raise ValueError("No non-valid images found for shuffling!")
    
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]
    
    # Log valid (unchanged)
    valid_dir = os.path.join(dataset_path, "valid/images")
    valid_count = len(glob(os.path.join(valid_dir, "*.*"))) if os.path.exists(valid_dir) else 0
    print(f"Valid unchanged: {valid_count} images")
    
    # Prepare folders (train/test only; keep valid as-is)
    for folder in ["train/images", "train/labels", "test/images", "test/labels"]:
        os.makedirs(os.path.join(dataset_path, folder), exist_ok=True)
    
    def move_pair(img_path, dest_folder):
        label_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        img_dest = os.path.join(dataset_path, dest_folder, "images", os.path.basename(img_path))
        lbl_dest = os.path.join(dataset_path, dest_folder, "labels", os.path.basename(label_path))
        
        if os.path.abspath(img_path) != os.path.abspath(img_dest):
            shutil.move(img_path, img_dest)
            if os.path.exists(label_path):
                shutil.move(label_path, lbl_dest)
    
    # Move to train/test (no valid involved)
    for img in train_images:
        move_pair(img, "train")
    for img in test_images:
        move_pair(img, "test")
    
    print(f"✅ Shuffled! Train: {len(train_images)}, Test: {len(test_images)} (valid unchanged)")
    return len(train_images), len(test_images)