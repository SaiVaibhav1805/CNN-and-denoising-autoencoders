"""Dataset Downloader and Synthetic Starter Generator for Smart Waste Classification.

PRD Recommended Dataset:
- Primary: TrashNet (Stanford CS229 Project by Gary Thung & Mindy Yang)
  - 2,527 images across 6 classes: cardboard (403), glass (501), metal (410), paper (594), plastic (482), trash (137).
  - Repositories:
    * https://github.com/garythung/trashnet
    * https://github.com/Ebuka24Stephen/trashnet-dataset
    * Hugging Face: https://huggingface.co/datasets/TrashBusters/combined (Larger alternative)

This script provides:
1. Automated download of TrashNet zip from verified mirrors.
2. Extraction of local zip if the user downloads `dataset-resized.zip` manually.
3. Generation of a synthetic starter dataset for immediate pipeline smoke-testing.
"""

import argparse
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src import config

# Mirrors for TrashNet
TRASHNET_MIRRORS = [
    # Direct raw archive or resized mirror
    "https://github.com/Ebuka24Stephen/trashnet-dataset/archive/refs/heads/master.zip",
    "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
]


def generate_synthetic_waste_sample(class_name: str, seed: int, size=(256, 256)) -> Image.Image:
    """Generates a realistic synthetic waste sample with characteristic colors and textures."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    w, h = size
    
    if class_name == "cardboard":
        # Warm kraft paper / cardboard brown with corrugation ridges
        base_color = (rng.integers(180, 210), rng.integers(135, 165), rng.integers(90, 120))
        img = Image.new("RGB", size, color=base_color)
        draw = ImageDraw.Draw(img)
        for y in range(0, h, 12):
            draw.line([(0, y), (w, y)], fill=(base_color[0] - 25, base_color[1] - 20, base_color[2] - 15), width=3)
        draw.rectangle([w*0.15, h*0.2, w*0.85, h*0.8], outline=(120, 80, 40), width=4)
        
    elif class_name == "glass":
        # Translucent greenish / cyan reflective bottle contour
        img = Image.new("RGB", size, color=(245, 248, 250))
        draw = ImageDraw.Draw(img)
        glass_tint = (rng.integers(150, 190), rng.integers(220, 245), rng.integers(210, 235))
        # Draw bottle shape
        draw.ellipse([w*0.35, h*0.1, w*0.65, h*0.3], fill=glass_tint, outline=(100, 160, 160), width=3)
        draw.rounded_rectangle([w*0.3, h*0.25, w*0.7, h*0.85], radius=25, fill=glass_tint, outline=(80, 140, 140), width=4)
        # Specular reflection line
        draw.line([w*0.38, h*0.3, w*0.38, h*0.8], fill=(255, 255, 255), width=6)
        
    elif class_name == "metal":
        # Metallic silver-grey cylindrical aluminum can
        img = Image.new("RGB", size, color=(235, 235, 240))
        draw = ImageDraw.Draw(img)
        metal_color = (rng.integers(190, 215), rng.integers(195, 220), rng.integers(205, 230))
        draw.rounded_rectangle([w*0.25, h*0.2, w*0.75, h*0.8], radius=15, fill=metal_color, outline=(130, 135, 145), width=4)
        draw.ellipse([w*0.25, h*0.15, w*0.75, h*0.25], fill=(225, 230, 235), outline=(130, 135, 145), width=3)
        draw.ellipse([w*0.4, h*0.18, w*0.6, h*0.22], fill=(160, 165, 175), outline=(100, 105, 115), width=2)
        # Metallic sheen
        for x_pos in [0.45, 0.55]:
            draw.line([w*x_pos, h*0.25, w*x_pos, h*0.75], fill=(255, 255, 255), width=4)

    elif class_name == "paper":
        # White / off-white crumpled sheet with faint text lines
        img = Image.new("RGB", size, color=(250, 250, 252))
        draw = ImageDraw.Draw(img)
        sheet_color = (248, 248, 245)
        draw.polygon([(w*0.2, h*0.15), (w*0.8, h*0.18), (w*0.75, h*0.85), (w*0.18, h*0.8)], fill=sheet_color, outline=(190, 190, 195))
        # Ruled text lines
        for y_offset in range(int(h*0.28), int(h*0.78), 16):
            draw.line([w*0.26, y_offset, w*0.7, y_offset], fill=(rng.integers(90, 140), rng.integers(110, 150), rng.integers(180, 220)), width=2)
            
    elif class_name == "plastic":
        # Bright colored plastic bottle / cup
        img = Image.new("RGB", size, color=(242, 242, 245))
        draw = ImageDraw.Draw(img)
        plastic_colors = [
            (240, 80, 60),    # Red
            (50, 140, 240),   # Blue
            (245, 170, 30),   # Yellow/Orange
            (60, 200, 120)    # Green
        ]
        color_choice = plastic_colors[rng.integers(0, len(plastic_colors))]
        # Bottle neck and body
        draw.rectangle([w*0.42, h*0.15, w*0.58, h*0.3], fill=color_choice, outline=(40, 40, 40), width=2)
        draw.ellipse([w*0.28, h*0.28, w*0.72, h*0.82], fill=color_choice, outline=(40, 40, 40), width=3)
        draw.rectangle([w*0.4, h*0.12, w*0.6, h*0.16], fill=(50, 50, 50))  # Cap
        
    elif class_name == "trash":
        # Mixed non-recyclable composite debris (wrappers, food container debris)
        img = Image.new("RGB", size, color=(230, 228, 225))
        draw = ImageDraw.Draw(img)
        for _ in range(8):
            poly_points = [
                (rng.integers(int(w*0.2), int(w*0.8)), rng.integers(int(h*0.2), int(h*0.8)))
                for _ in range(rng.integers(3, 6))
            ]
            fill_c = (rng.integers(80, 180), rng.integers(60, 140), rng.integers(50, 120))
            draw.polygon(poly_points, fill=fill_c, outline=(60, 60, 60))
            
    # Apply subtle blur to soften edges for realistic camera feel
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def create_starter_dataset(samples_per_class: int = 40):
    """Generates balanced synthetic starter images across all 6 classes."""
    print(f"Generating synthetic starter dataset ({samples_per_class} images/class across {len(config.CLASSES)} classes)...")
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    total_created = 0
    for cls_name in config.CLASSES:
        cls_dir = config.RAW_DATA_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in range(samples_per_class):
            img = generate_synthetic_waste_sample(cls_name, seed=(idx * 17 + hash(cls_name) % 1000))
            save_path = cls_dir / f"{cls_name}_{idx+1:03d}.jpg"
            img.save(save_path, "JPEG", quality=90)
            total_created += 1
            
    print(f"[OK] Generated {total_created} starter images in {config.RAW_DATA_DIR}")
    print("Classes generated:")
    for cls_name in config.CLASSES:
        cnt = len(list((config.RAW_DATA_DIR / cls_name).glob("*.jpg")))
        print(f"  - {cls_name:10s}: {cnt:3d} images")


def extract_local_trashnet_zip(zip_path: str):
    """Extracts an existing TrashNet zip archive directly into data/raw."""
    zip_p = Path(zip_path)
    if not zip_p.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
        
    print(f"Extracting {zip_path} to {config.RAW_DATA_DIR}...")
    temp_dir = config.DATA_DIR / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_p, 'r') as zf:
        zf.extractall(temp_dir)
        
    # Search for class subdirectories (cardboard, glass, metal, paper, plastic, trash)
    found_classes = 0
    for root, dirs, _ in os.walk(temp_dir):
        for d in dirs:
            if d.lower() in config.CLASSES:
                dest = config.RAW_DATA_DIR / d.lower()
                dest.mkdir(parents=True, exist_ok=True)
                src = Path(root) / d
                for f in src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dest / f.name)
                found_classes += 1
                
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[OK] Successfully extracted {found_classes} classes to {config.RAW_DATA_DIR}")


def download_from_huggingface(repo_id: str = "garythung/trashnet"):
    """Downloads official TrashNet dataset directly from Hugging Face hub (garythung/trashnet)."""
    print(f"Downloading official dataset from Hugging Face ({repo_id})...")
    from huggingface_hub import hf_hub_download
    
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename="dataset-resized.zip",
        repo_type="dataset"
    )
    print(f"Downloaded official archive to: {zip_path}")
    
    # Extract into data/raw
    raw_dir = config.RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_counts = {c: 0 for c in config.CLASSES}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            if len(parts) >= 3 and parts[0] == "dataset-resized" and not member.endswith("/"):
                cls_name = parts[1].lower()
                if cls_name in config.CLASSES:
                    dest_dir = raw_dir / cls_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / parts[-1]
                    with zf.open(member) as src, open(dest_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted_counts[cls_name] += 1
                    
    print("[OK] Successfully unpacked official TrashNet dataset to:", raw_dir)
    for c, cnt in extracted_counts.items():
        print(f"  - {c:10s}: {cnt:4d} images")
    return zip_path


def load_trashnet_hf_dataset():
    """Provides a native Hugging Face datasets.Dataset object for garythung/trashnet.
    
    Usage:
        from src.dataset_downloader import load_trashnet_hf_dataset
        ds = load_trashnet_hf_dataset()
        print(ds['train'][0])
    """
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, ClassLabel
    from src.prepare_data import load_split_manifest
    
    # Ensure splits exist
    train_manifest = config.SPLITS_DIR / "train.json"
    if not train_manifest.exists():
        from src.prepare_data import run_pipeline
        run_pipeline()
        
    features = Features({
        "image": HFImage(),
        "label": ClassLabel(names=config.CLASSES)
    })
    
    splits = {}
    for split_name in ["train", "val", "test"]:
        records = load_split_manifest(split_name)
        data = {
            "image": [r["path"] for r in records],
            "label": [r["label"] for r in records]
        }
        splits[split_name] = Dataset.from_dict(data, features=features)
        
    return DatasetDict(splits)


def download_trashnet(target_zip: Path = config.DATA_DIR / "trashnet.zip"):
    """Downloads TrashNet dataset archive from available mirrors or Hugging Face."""
    print("Initiating TrashNet download...")
    try:
        download_from_huggingface()
        return
    except Exception as e:
        print(f"Hugging Face hub download encountered: {e}. Trying secondary mirrors...")
        
    headers = {"User-Agent": "Mozilla/5.0"}
    success = False
    for url in TRASHNET_MIRRORS:
        try:
            print(f"Attempting download from: {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(target_zip, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"Downloaded archive to {target_zip}")
            extract_local_trashnet_zip(str(target_zip))
            success = True
            break
        except Exception as e:
            print(f"Mirror failed ({e}). Trying next...")
            
    if not success:
        print("[!] Automated download was unavailable. Falling back to synthetic starter dataset.")
        create_starter_dataset(samples_per_class=40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset setup utility for Smart Waste Classification")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic starter dataset")
    parser.add_argument("--download", action="store_true", help="Download TrashNet from online repository")
    parser.add_argument("--hf", action="store_true", help="Download official TrashNet directly from Hugging Face")
    parser.add_argument("--zip", type=str, help="Path to local dataset-resized.zip file")
    parser.add_argument("--samples", type=int, default=40, help="Number of samples per class for mock generator")
    
    args = parser.parse_args()
    
    if args.hf:
        download_from_huggingface()
    elif args.zip:
        extract_local_trashnet_zip(args.zip)
    elif args.download:
        download_trashnet()
    else:
        # Default behavior: generate starter dataset if directory is empty
        create_starter_dataset(samples_per_class=args.samples)
