"""CNN Classifier Training Module.

Satisfies PRD Functional Requirements:
- FR-04: Train a baseline CNN classifier using class labels.
- FR-06: Save trained model weights to models/cnn_classifier.keras.
- Section 14 Mitigation: Class weighting to handle imbalance + Data Augmentation.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import config
from src.models import build_cnn_classifier
from src.prepare_data import (
    load_dataset_arrays,
    get_class_weights_dict,
    create_tf_dataset_for_cnn,
    run_pipeline
)


def plot_training_history(history, save_path=config.MODELS_DIR / "cnn_training_history.png"):
    """Plots training and validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy", lw=2, color="#2563eb")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy", lw=2, color="#10b981", linestyle="--")
    ax1.set_title("CNN Classifier Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")
    
    # Loss
    ax2.plot(history.history["loss"], label="Train Loss", lw=2, color="#ef4444")
    ax2.plot(history.history["val_loss"], label="Val Loss", lw=2, color="#f59e0b", linestyle="--")
    ax2.set_title("CNN Classifier Loss (Sparse Categorical CE)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Training plots saved to {save_path}")


def train_cnn(epochs: int = config.CNN_EPOCHS, batch_size: int = config.BATCH_SIZE):
    """Trains the CNN classifier on the prepared train split, validated on val split."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    
    # Ensure splits exist
    if not (config.SPLITS_DIR / "train.json").exists():
        print("[!] Splits not found. Running prepare_data pipeline first...")
        run_pipeline()
        
    print("=== [Phase 2] Training Baseline CNN Classifier ===")
    print(f"Target classes: {config.CLASSES}")
    
    X_train, y_train = load_dataset_arrays("train")
    X_val, y_val = load_dataset_arrays("val")
    
    print(f"Loaded {len(X_train)} training images, {len(X_val)} validation images.")
    
    # Compute balanced class weights to address TrashNet class imbalance (FR-04, Risk Mitigation)
    class_weights = get_class_weights_dict(y_train)
    print("Computed class weights for balanced loss:")
    for idx, w in class_weights.items():
        print(f"  - {config.IDX_TO_CLASS[idx]:10s}: {w:.3f}")
        
    # Build model
    model = build_cnn_classifier(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        dropout_rate=0.3,
        learning_rate=config.CNN_LEARNING_RATE
    )
    model.summary()
    
    # Prepare TensorFlow datasets
    train_dataset = create_tf_dataset_for_cnn(X_train, y_train, batch_size=batch_size, augment=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.CNN_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\nStarting CNN model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Ensure final best model is saved
    model.save(config.CNN_MODEL_PATH)
    print(f"[OK] Best CNN model saved to: {config.CNN_MODEL_PATH}")
    
    plot_training_history(history)
    
    val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%, Validation Loss: {val_loss:.4f}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline CNN classifier")
    parser.add_argument("--epochs", type=int, default=config.CNN_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    
    train_cnn(epochs=args.epochs, batch_size=args.batch_size)
