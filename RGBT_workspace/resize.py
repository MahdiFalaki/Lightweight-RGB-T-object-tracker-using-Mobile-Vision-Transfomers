import cv2
import os
from tqdm import tqdm

import cv2
import os
from tqdm import tqdm

def batch_resize_ir_to_match_rgb(rgb_dir, ir_dir, save_dir=None):
    os.makedirs(save_dir or ir_dir, exist_ok=True)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png', '.bmp'))])
    if not rgb_files:
        raise RuntimeError(f"No image files found in {rgb_dir}")

    rgb_sample_path = os.path.join(rgb_dir, rgb_files[0])
    rgb_sample = cv2.imread(rgb_sample_path)
    if rgb_sample is None:
        raise RuntimeError(f"Could not read sample RGB image: {rgb_sample_path}")

    target_size = (rgb_sample.shape[1], rgb_sample.shape[0])  # (width, height)

    ir_images = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.jpg', '.png', '.bmp'))])
    for fname in tqdm(ir_images, desc="Resizing IR images"):
        ir_path = os.path.join(ir_dir, fname)
        img = cv2.imread(ir_path)
        if img is None:
            print(f"[WARNING] Skipping unreadable file: {ir_path}")
            continue
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        save_path = os.path.join(save_dir or ir_dir, fname)
        cv2.imwrite(save_path, resized)

batch_resize_ir_to_match_rgb('/home/mark/Codes/mahdi_codes_folder/datasets/GTOT/occBike/v', '/home/mark/Codes/mahdi_codes_folder/datasets/GTOT/occBike/i')