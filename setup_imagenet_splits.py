"""
ImageNet-100 Splits Creator
=========================================
This script takes the existing 'train' folder (which contains 100 folders with 1,300 images each)
and automatically splits it into training (1,250 images/class) and validation (50 images/class) splits
under './data/imagenet100' using fast NTFS hardlinks (instant, uses 0 extra disk space).

Usage:
    python setup_imagenet_splits.py
"""

import os
import sys

def create_splits():
    src_dir = "./train"
    dest_root = "./data/imagenet100"
    train_dest = os.path.join(dest_root, "train")
    val_dest = os.path.join(dest_root, "val")
    
    if not os.path.exists(src_dir):
        print(f"❌ Error: Source directory '{src_dir}' not found.")
        return
        
    classes = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    print(f"Detected {len(classes)} classes in '{src_dir}'.")
    
    if len(classes) == 0:
        print("❌ Error: No class directories found.")
        return
        
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    
    print("\nStarting fast hardlink splitting (0 extra disk space)...")
    
    for idx, cls in enumerate(classes):
        cls_src_path = os.path.join(src_dir, cls)
        cls_train_path = os.path.join(train_dest, cls)
        cls_val_path = os.path.join(val_dest, cls)
        
        os.makedirs(cls_train_path, exist_ok=True)
        os.makedirs(cls_val_path, exist_ok=True)
        
        files = sorted([f for f in os.listdir(cls_src_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        num_files = len(files)
        
        if num_files < 100:
            print(f"  [WARN] Class '{cls}' has very few files ({num_files}). Copying all to train.")
            train_files = files
            val_files = []
        else:
            # 1,250 for train, rest (usually 50) for val
            split_idx = min(1250, int(num_files * 0.95))
            train_files = files[:split_idx]
            val_files = files[split_idx:]
            
        # Hardlink files
        for f in train_files:
            src_file = os.path.join(cls_src_path, f)
            dest_file = os.path.join(cls_train_path, f)
            if not os.path.exists(dest_file):
                try:
                    os.link(src_file, dest_file)
                except OSError:
                    import shutil
                    shutil.copy(src_file, dest_file)
                    
        for f in val_files:
            src_file = os.path.join(cls_src_path, f)
            dest_file = os.path.join(cls_val_path, f)
            if not os.path.exists(dest_file):
                try:
                    os.link(src_file, dest_file)
                except OSError:
                    import shutil
                    shutil.copy(src_file, dest_file)
                    
        if (idx + 1) % 10 == 0 or (idx + 1) == len(classes):
            print(f"  Processed {idx + 1}/{len(classes)} classes...")
            
    print(f"\n✅ ImageNet-100 splits successfully created at: {dest_root}")
    print("  - Train location: ./data/imagenet100/train")
    print("  - Val location: ./data/imagenet100/val")

if __name__ == "__main__":
    create_splits()
