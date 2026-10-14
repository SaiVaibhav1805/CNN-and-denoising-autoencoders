"""Inference and Prediction Pipeline.

Accepts an input image, optionally introduces noise, applies Denoising Autoencoder,
and classifies with the CNN Classifier.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from src import config
from src.prepare_data import load_and_preprocess_image, add_gaussian_noise


class WastePredictor:
    """End-to-end pipeline predictor for Waste Classification and Denoising."""
    
    def __init__(
        self,
        cnn_path: Union[str, Path] = config.CNN_MODEL_PATH,
        dae_path: Union[str, Path] = config.DAE_MODEL_PATH
    ):
        self.cnn_path = Path(cnn_path)
        self.dae_path = Path(dae_path)
        self.cnn_model = None
        self.dae_model = None
        self._load_models()
        
    def _load_models(self):
        if not self.cnn_path.exists():
            raise FileNotFoundError(f"CNN model not found at {self.cnn_path}. Train the CNN first.")
        if not self.dae_path.exists():
            raise FileNotFoundError(f"DAE model not found at {self.dae_path}. Train the DAE first.")
            
        print(f"Loading CNN from {self.cnn_path}...")
        self.cnn_model = keras.models.load_model(self.cnn_path)
        print(f"Loading DAE from {self.dae_path}...")
        self.dae_model = keras.models.load_model(self.dae_path)
        
    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        noise_factor: float = 0.0
    ) -> Dict[str, Any]:
        """Runs end-to-end inference on a single image.
        
        Args:
            image_input: File path, PIL Image, or float array [0, 1].
            noise_factor: Standard deviation of Gaussian noise to inject (0.0 for clean).
            
        Returns:
            Dictionary containing clean, noisy, denoised images, class predictions,
            and confidence scores.
        """
        # Load and normalize
        if isinstance(image_input, (str, Path)):
            clean_arr = load_and_preprocess_image(str(image_input))
        elif isinstance(image_input, Image.Image):
            resized = image_input.convert("RGB").resize(
                (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), Image.Resampling.BILINEAR
            )
            clean_arr = np.array(resized, dtype=np.float32) / 255.0
        elif isinstance(image_input, np.ndarray):
            clean_arr = image_input.astype(np.float32)
            if clean_arr.max() > 1.0:
                clean_arr /= 255.0
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
        # Add batch dimension: (1, 128, 128, 3)
        batch_clean = np.expand_dims(clean_arr, axis=0)
        
        # Corrupt if noise_factor > 0
        if noise_factor > 0:
            batch_noisy = add_gaussian_noise(batch_clean, noise_factor=noise_factor)
        else:
            batch_noisy = batch_clean.copy()
            
        # Run DAE Reconstruction
        batch_denoised = self.dae_model.predict(batch_noisy, verbose=0)
        batch_denoised = np.clip(batch_denoised, 0.0, 1.0)
        
        # Run CNN Classification on all three
        probs_clean = self.cnn_model.predict(batch_clean, verbose=0)[0]
        probs_noisy = self.cnn_model.predict(batch_noisy, verbose=0)[0]
        probs_denoised = self.cnn_model.predict(batch_denoised, verbose=0)[0]
        
        pred_clean_idx = int(np.argmax(probs_clean))
        pred_noisy_idx = int(np.argmax(probs_noisy))
        pred_denoised_idx = int(np.argmax(probs_denoised))
        
        # Compute reconstruction MSE and PSNR
        mse = float(np.mean((batch_clean[0] - batch_denoised[0]) ** 2))
        psnr = float(20 * np.log10(1.0 / np.sqrt(mse))) if mse > 0 else 100.0
        
        return {
            "clean_image": batch_clean[0],
            "noisy_image": batch_noisy[0],
            "denoised_image": batch_denoised[0],
            "noise_factor": noise_factor,
            "reconstruction_metrics": {
                "mse": mse,
                "psnr": psnr
            },
            "clean_prediction": {
                "class_name": config.IDX_TO_CLASS[pred_clean_idx],
                "confidence": float(probs_clean[pred_clean_idx]),
                "probabilities": {config.IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs_clean)}
            },
            "noisy_prediction": {
                "class_name": config.IDX_TO_CLASS[pred_noisy_idx],
                "confidence": float(probs_noisy[pred_noisy_idx]),
                "probabilities": {config.IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs_noisy)}
            },
            "denoised_prediction": {
                "class_name": config.IDX_TO_CLASS[pred_denoised_idx],
                "confidence": float(probs_denoised[pred_denoised_idx]),
                "probabilities": {config.IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs_denoised)}
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Run Smart Waste Classification inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input waste image")
    parser.add_argument("--noise", type=float, default=0.25, help="Simulated Gaussian noise level (e.g. 0.25)")
    args = parser.parse_args()
    
    predictor = WastePredictor()
    results = predictor.predict_image(args.image, noise_factor=args.noise)
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Input Image: {args.image}")
    print(f"Simulated Noise: {args.noise}")
    print(f"Reconstruction PSNR: {results['reconstruction_metrics']['psnr']:.2f} dB, MSE: {results['reconstruction_metrics']['mse']:.5f}")
    print("-" * 50)
    
    c_pred = results["clean_prediction"]
    n_pred = results["noisy_prediction"]
    d_pred = results["denoised_prediction"]
    
    print(f"1. Clean Input:    Class = {c_pred['class_name']:<10} (Confidence: {c_pred['confidence']*100:.1f}%)")
    print(f"2. Noisy Input:    Class = {n_pred['class_name']:<10} (Confidence: {n_pred['confidence']*100:.1f}%)")
    print(f"3. Denoised Input: Class = {d_pred['class_name']:<10} (Confidence: {d_pred['confidence']*100:.1f}%)")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
