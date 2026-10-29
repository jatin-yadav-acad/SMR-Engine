"""
ImageNet & TinyImageNet Dataset Setup Utility
===================================================
Run this script to automatically:
1. Download and extract TinyImageNet-200 (120MB CS231n subset).
2. Extract the ImageNet-100 subset from your local ImageNet-1K installation.

Usage:
    python setup_datasets.py --tiny
    python setup_datasets.py --imagenet1k /path/to/imagenet1k
"""

import os
import sys
import shutil
import random
import zipfile
import urllib.request
import argparse

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, float(read_so_far) * 100 / total_size)
        sys.stdout.write(f"\r  Downloading: {percent:.1f}% [{read_so_far // (1024*1024)}MB / {total_size // (1024*1024)}MB]")
    else:
        sys.stdout.write(f"\r  Downloading: {read_so_far // (1024*1024)}MB")
    sys.stdout.flush()

def setup_tiny_imagenet(data_dir):
    print("\n" + "="*50)
    print("SETTING UP TINY-IMAGENET-200 (CS231n)")
    print("="*50)
    
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_dest = os.path.join(data_dir, "tiny-imagenet-200")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if os.path.exists(extract_dest):
        print(f"[INFO] TinyImageNet already exists at {extract_dest}. Skipping download.")
        return
        
    print(f"Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, zip_path, download_progress)
        print("\n[SUCCESS] Download completed. Extracting archive...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Clean up zip file
        os.remove(zip_path)
        print(f"✅ TinyImageNet successfully set up at: {extract_dest}")
    except Exception as e:
        print(f"\n❌ Error downloading/extracting TinyImageNet: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

def setup_imagenet_100(imagenet_1k_root, output_dir, seed=42):
    print("\n" + "="*50)
    print("BUILDING IMAGENET-100 FROM LOCAL IMAGENET-1K")
    print("="*50)
    
    train_src = os.path.join(imagenet_1k_root, "train")
    val_src = os.path.join(imagenet_1k_root, "val")
    
    if not os.path.exists(train_src) or not os.path.exists(val_src):
        print(f"❌ Error: Could not find train/val folders in local path: {imagenet_1k_root}")
        print("Please verify your ImageNet-1K path contains both 'train' and 'val' subfolders.")
        return
        
    train_dst = os.path.join(output_dir, "train")
    val_dst = os.path.join(output_dir, "val")
    
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)
    
    # Get sorted classes list
    classes = sorted(os.listdir(train_src))
    print(f"Total classes found in ImageNet-1K: {len(classes)}")
    
    if len(classes) < 100:
        print("❌ Error: ImageNet-1K source directory contains fewer than 100 classes.")
        return
        
    # Sample exactly 100 classes with exact seed for reproducibility
    random.seed(seed)
    selected_classes = random.sample(classes, 100)
    print(f"Randomly selected 100 classes under seed={seed}.")
    
    def copy_subset(src_root, dst_root):
        for idx, cls in enumerate(selected_classes):
            src_cls_path = os.path.join(src_root, cls)
            dst_cls_path = os.path.join(dst_root, cls)
            
            if not os.path.exists(src_cls_path):
                print(f"  [WARN] Class folder {cls} not found in {src_root}. Skipping.")
                continue
                
            shutil.copytree(src_cls_path, dst_cls_path, dirs_exist_ok=True)
            if (idx + 1) % 10 == 0 or (idx + 1) == 100:
                print(f"  Processed {idx + 1}/100 classes...")
                
    print("\n1. Copying Train subset...")
    copy_subset(train_src, train_dst)
    
    print("\n2. Copying Val/Test subset...")
    copy_subset(val_src, val_dst)
    
    print(f"\n✅ ImageNet-100 successfully built and saved at: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="ImageNet & TinyImageNet Dataset Setup Utility")
    parser.add_argument("--tiny", action="store_true", help="Download and set up TinyImageNet-200")
    parser.add_argument("--imagenet1k", type=str, help="Absolute path to your local ImageNet-1K installation root")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where datasets are stored")
    args = parser.parse_args()
    
    if not args.tiny and not args.imagenet1k:
        parser.print_help()
        print("\nExample commands:")
        print("  python setup_datasets.py --tiny")
        print("  python setup_datasets.py --imagenet1k D:\\datasets\\imagenet-1k")
        return
        
    if args.tiny:
        setup_tiny_imagenet(args.data_dir)
        
    if args.imagenet1k:
        output_dir = os.path.join(args.data_dir, "imagenet100")
        setup_imagenet_100(args.imagenet1k, output_dir)

if __name__ == "__main__":
    main()
