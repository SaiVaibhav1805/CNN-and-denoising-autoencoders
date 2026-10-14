"""Convolutional Denoising Autoencoder (DAE) Training Module.

Satisfies PRD Functional Requirements:
- FR-05: Create noisy inputs from clean training images and train DAE to reconstruct clean targets.
- FR-06: Save trained model weights to models/denoising_autoencoder.keras & models/encoder.keras.
- FR-07: Display clean, noisy, and reconstructed images side-by-side with PSNR & MSE.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import config
from src.models import build_denoising_autoencoder
from src.prepare_data import (
    load_dataset_arrays,
    add_gaussian_noise,
    create_tf_dataset_for_dae,
    run_pipeline
)


def calculate_psnr(clean_img: np.ndarray, recon_img: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) between clean and reconstructed images."""
    mse = np.mean((clean_img - recon_img) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 1.0  # Normalized [0, 1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def visualize_reconstructions(
    clean_images: np.ndarray,
    noisy_images: np.ndarray,
    reconstructed_images: np.ndarray,
    num_samples: int = 5,
    save_path=config.MODELS_DIR / "dae_sample_reconstruction.png"
):
    """Saves a multi-row visual comparison: Clean vs Noisy vs Reconstructed (FR-07)."""
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 8))
    
    for i in range(num_samples):
        # Clean
        axes[0, i].imshow(clean_images[i])
        axes[0, i].set_title(f"Clean Target #{i+1}", fontsize=11, fontweight="bold")
        axes[0, i].axis("off")
        
        # Noisy
        axes[1, i].imshow(noisy_images[i])
        noisy_psnr = calculate_psnr(clean_images[i], noisy_images[i])
        axes[1, i].set_title(f"Noisy Input\nPSNR: {noisy_psnr:.1f} dB", fontsize=10, color="#b91c1c")
        axes[1, i].axis("off")
        
        # Reconstructed
        axes[2, i].imshow(np.clip(reconstructed_images[i], 0.0, 1.0))
        recon_psnr = calculate_psnr(clean_images[i], reconstructed_images[i])
        recon_mse = np.mean((clean_images[i] - reconstructed_images[i]) ** 2)
        axes[2, i].set_title(f"DAE Denoised\nPSNR: {recon_psnr:.1f} dB\nMSE: {recon_mse:.4f}", fontsize=10, color="#15803d")
        axes[2, i].axis("off")
        
    plt.suptitle("Convolutional Denoising Autoencoder: Clean vs Noisy vs Denoised Reconstructions", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] DAE Reconstruction comparison plot saved to: {save_path}")


def train_dae(
    epochs: int = config.DAE_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    noise_factor: float = config.NOISE_FACTOR
):
    """Trains DAE to denoise corrupted waste images."""
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    
    if not (config.SPLITS_DIR / "train.json").exists():
        print("[!] Splits not found. Running prepare_data pipeline first...")
        run_pipeline()
        
    print("=== [Phase 3] Building and Training Denoising Autoencoder ===")
    print(f"Noise corruption: Gaussian with standard deviation factor = {noise_factor}")
    
    X_train_clean, _ = load_dataset_arrays("train")
    X_val_clean, _ = load_dataset_arrays("val")
    
    # Generate noisy validation pairs
    X_val_noisy = add_gaussian_noise(X_val_clean, noise_factor=noise_factor, seed=config.RANDOM_SEED)
    
    print(f"Loaded {len(X_train_clean)} clean training images, {len(X_val_clean)} clean validation images.")
    
    # Build models
    autoencoder, encoder = build_denoising_autoencoder(
        input_shape=config.INPUT_SHAPE,
        learning_rate=config.DAE_LEARNING_RATE
    )
    autoencoder.summary()
    
    # Prepare datasets
    train_dataset = create_tf_dataset_for_dae(X_train_clean, batch_size=batch_size, noise_factor=noise_factor, shuffle=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_noisy, X_val_clean)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
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
            filepath=str(config.DAE_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\nStarting Denoising Autoencoder training...")
    history = autoencoder.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save both DAE and the isolated Encoder
    autoencoder.save(config.DAE_MODEL_PATH)
    encoder.save(config.ENCODER_MODEL_PATH)
    print(f"[OK] Full DAE saved to: {config.DAE_MODEL_PATH}")
    print(f"[OK] Encoder model saved to: {config.ENCODER_MODEL_PATH}")
    
    # Evaluate sample reconstructions (FR-07)
    sample_indices = np.random.choice(len(X_val_clean), min(5, len(X_val_clean)), replace=False)
    sample_clean = X_val_clean[sample_indices]
    sample_noisy = X_val_noisy[sample_indices]
    sample_reconstructed = autoencoder.predict(sample_noisy, verbose=0)
    
    visualize_reconstructions(sample_clean, sample_noisy, sample_reconstructed)
    
    val_loss, val_mae = autoencoder.evaluate(val_dataset, verbose=0)
    avg_psnr = np.mean([calculate_psnr(c, r) for c, r in zip(sample_clean, sample_reconstructed)])
    print(f"\nFinal Validation MSE Loss: {val_loss:.5f}, MAE: {val_mae:.5f}")
    print(f"Sample Denoised Average PSNR: {avg_psnr:.2f} dB")
    
    return autoencoder, encoder, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Convolutional Denoising Autoencoder")
    parser.add_argument("--epochs", type=int, default=config.DAE_EPOCHS, help="Number of DAE training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--noise_factor", type=float, default=config.NOISE_FACTOR, help="Noise standard deviation factor")
    args = parser.parse_args()
    
    train_dae(epochs=args.epochs, batch_size=args.batch_size, noise_factor=args.noise_factor)
