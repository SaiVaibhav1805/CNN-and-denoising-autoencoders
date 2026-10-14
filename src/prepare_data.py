"""Data Preparation, Splitting, Normalization, and Noise Generation Pipeline.

Satisfies PRD Functional Requirements:
- FR-01: Load image dataset from class-specific folders.
- FR-02: Resize images to 128x128 and normalize pixels to [0, 1].
- FR-03: Split images into reproducible train (70%), val (15%), test (15%) sets.
- FR-05: Create noisy inputs from clean images for Denoising Autoencoder training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers

from src import config


def get_image_paths_and_labels(raw_dir: Path = config.RAW_DATA_DIR) -> Tuple[List[str], List[int]]:
    """Discovers all images in class subdirectories and assigns corresponding integer labels."""
    image_paths: List[str] = []
    labels: List[int] = []
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for class_name in config.CLASSES:
        class_folder = raw_dir / class_name
        if not class_folder.exists() or not class_folder.is_dir():
            continue
        
        class_idx = config.CLASS_TO_IDX[class_name]
        for file_path in class_folder.iterdir():
            if file_path.suffix.lower() in valid_exts:
                image_paths.append(str(file_path.resolve()))
                labels.append(class_idx)
                
    return image_paths, labels


def create_and_save_splits(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED
) -> Dict[str, List[dict]]:
    """Performs stratified train/val/test split and saves split manifests to JSON files."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    if len(image_paths) == 0:
        raise ValueError("No images found in raw directory. Please download or generate dataset first.")
        
    indices = np.arange(len(image_paths))
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, val_test_idx, y_train, y_val_test = train_test_split(
        indices,
        labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # Second split: val vs test
    val_relative_ratio = val_ratio / val_test_ratio
    val_idx, test_idx, y_val, y_test = train_test_split(
        val_test_idx,
        y_val_test,
        test_size=(1.0 - val_relative_ratio),
        stratify=y_val_test,
        random_state=seed
    )
    
    def make_records(idxs):
        return [
            {
                "path": image_paths[i],
                "label": int(labels[i]),
                "class_name": config.IDX_TO_CLASS[labels[i]]
            }
            for i in idxs
        ]
        
    splits = {
        "train": make_records(train_idx),
        "val": make_records(val_idx),
        "test": make_records(test_idx),
    }
    
    # Save splits to JSON manifests
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        out_path = config.SPLITS_DIR / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
            
    print(f"[OK] Data split completed: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test images.")
    return splits


def load_split_manifest(split_name: str) -> List[dict]:
    """Loads a split manifest from data/splits/<split_name>.json."""
    split_file = config.SPLITS_DIR / f"{split_name}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run prepare_data first.")
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
) -> np.ndarray:
    """Reads image from disk, converts to RGB, resizes to target_size, and normalizes to [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def add_gaussian_noise(
    images: np.ndarray,
    noise_factor: float = config.NOISE_FACTOR,
    seed: int = None
) -> np.ndarray:
    """Adds zero-mean Gaussian noise to images and clips pixel values to [0, 1].
    
    images: float array with values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_factor, size=images.shape)
    noisy_images = np.clip(images + noise, 0.0, 1.0).astype(np.float32)
    return noisy_images


def add_salt_and_pepper_noise(
    images: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: int = None
) -> np.ndarray:
    """Adds salt and pepper noise to images and keeps values in [0, 1]."""
    rng = np.random.default_rng(seed)
    noisy_images = images.copy()
    
    # Salt
    num_salt = np.ceil(amount * images.size * salt_vs_pepper)
    coords = [rng.integers(0, i - 1, int(num_salt)) for i in images.shape]
    noisy_images[tuple(coords)] = 1.0
    
    # Pepper
    num_pepper = np.ceil(amount * images.size * (1.0 - salt_vs_pepper))
    coords = [rng.integers(0, i - 1, int(num_pepper)) for i in images.shape]
    noisy_images[tuple(coords)] = 0.0
    
    return np.clip(noisy_images, 0.0, 1.0).astype(np.float32)


def load_dataset_arrays(split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads all images and integer labels for a split into memory as numpy arrays."""
    records = load_split_manifest(split_name)
    images = []
    labels = []
    for r in records:
        try:
            img = load_and_preprocess_image(r["path"])
            images.append(img)
            labels.append(r["label"])
        except Exception as e:
            print(f"[Warning] Failed to load {r['path']}: {e}")
            
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def get_class_weights_dict(labels: np.ndarray) -> Dict[int, float]:
    """Computes balanced class weights to address TrashNet class imbalance (PRD Risk Mitigation)."""
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    return weight_dict


def create_tf_dataset_for_cnn(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
    augment: bool = False
) -> tf.data.Dataset:
    """Creates an optimized tf.data.Dataset pipeline with optional data augmentation for CNN."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if augment:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
    dataset = dataset.shuffle(buffer_size=1024, seed=config.RANDOM_SEED)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_tf_dataset_for_dae(
    clean_images: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
    noise_factor: float = config.NOISE_FACTOR,
    shuffle: bool = True
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset pipeline yielding (noisy_image, clean_image) pairs for DAE."""
    noisy_images = add_gaussian_noise(clean_images, noise_factor=noise_factor)
    dataset = tf.data.Dataset.from_tensor_slices((noisy_images, clean_images))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024, seed=config.RANDOM_SEED)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def run_pipeline():
    """Main execution entrypoint for data preparation."""
    print("=== [Phase 1] Smart Waste Classification Data Preparation ===")
    image_paths, labels = get_image_paths_and_labels()
    
    if not image_paths:
        print(f"[!] No images found in {config.RAW_DATA_DIR}.")
        print("[!] Tip: Run 'python -m src.dataset_downloader --mock' to generate sample images")
        print("[!] Or run 'python -m src.dataset_downloader --download' to fetch TrashNet.")
        return
        
    print(f"Found {len(image_paths)} total images across {len(config.CLASSES)} classes.")
    
    # Class distribution
    for cls_name in config.CLASSES:
        idx = config.CLASS_TO_IDX[cls_name]
        cnt = labels.count(idx)
        print(f"  - {cls_name:10s}: {cnt:4d} images ({cnt/len(labels)*100:5.1f}%)")
        
    splits = create_and_save_splits(image_paths, labels)
    print("Data preparation finished successfully.")


if __name__ == "__main__":
    from tensorflow.keras import layers
    run_pipeline()
