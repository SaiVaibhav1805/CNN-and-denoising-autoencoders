"""Neural Network Architectures for Smart Waste Classification.

Includes:
1. Baseline CNN Classifier (PRD 8.1)
2. Convolutional Denoising Autoencoder (DAE) and Encoder submodel (PRD 8.2)
3. Optional Pretrained Encoder-based Classifier (PRD 8.3 Extension)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from src import config


def build_cnn_classifier(
    input_shape=config.INPUT_SHAPE,
    num_classes=config.NUM_CLASSES,
    dropout_rate=0.3,
    learning_rate=config.CNN_LEARNING_RATE
) -> Model:
    """Builds a robust Convolutional Neural Network for 6-class waste classification.
    
    Architecture:
        Input -> [Conv2D -> BN -> ReLU -> Conv2D -> ReLU -> MaxPool] x 3
              -> GlobalAveragePooling2D -> Dense(128) -> Dropout -> Softmax(6)
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Block 1: 128x128 -> 64x64
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # Block 2: 64x64 -> 32x32
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Block 3: 32x32 -> 16x16
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="WasteCNNClassifier")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_denoising_autoencoder(
    input_shape=config.INPUT_SHAPE,
    learning_rate=config.DAE_LEARNING_RATE
) -> tuple[Model, Model]:
    """Builds a Convolutional Denoising Autoencoder (DAE) and extracts the Encoder submodel.
    
    Architecture:
        Encoder:
          128x128x3 -> Conv(32) -> Pool -> 64x64x32
                    -> Conv(64) -> Pool -> 32x32x64
                    -> Conv(128) -> Pool -> 16x16x128 (Latent Bottleneck)
        Decoder:
          16x16x128 -> ConvTranspose(128) -> 32x32x128
                    -> ConvTranspose(64)  -> 64x64x64
                    -> ConvTranspose(32)  -> 128x128x32
                    -> Conv2D(3, sigmoid) -> 128x128x3 (Reconstructed clean image)
    
    Returns:
        autoencoder: Full DAE model (noisy image -> reconstructed clean image)
        encoder: Encoder submodel (noisy image -> latent representation)
    """
    inputs = layers.Input(shape=input_shape, name="noisy_input")
    
    # --- ENCODER ---
    e = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(inputs)
    e = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(e)  # 64x64
    
    e = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(e)
    e = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(e)  # 32x32
    
    e = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv3")(e)
    latent_space = layers.MaxPooling2D((2, 2), padding="same", name="latent_bottleneck")(e)  # 16x16x128
    
    encoder = Model(inputs=inputs, outputs=latent_space, name="WasteDAEEncoder")
    
    # --- DECODER ---
    d = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same", name="dec_deconv1")(latent_space)  # 32x32
    d = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same", name="dec_deconv2")(d)              # 64x64
    d = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same", name="dec_deconv3")(d)              # 128x128
    
    reconstructed = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="reconstructed_clean")(d)
    
    autoencoder = Model(inputs=inputs, outputs=reconstructed, name="WasteDenoisingAutoencoder")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )
    
    return autoencoder, encoder


def build_encoder_classifier(
    encoder_model: Model,
    num_classes=config.NUM_CLASSES,
    freeze_encoder: bool = True,
    learning_rate=config.CNN_LEARNING_RATE
) -> Model:
    """PRD Optional Extension: Reuse pretrained DAE encoder as feature extractor."""
    if freeze_encoder:
        encoder_model.trainable = False
        
    inputs = encoder_model.input
    latent = encoder_model.output
    
    x = layers.GlobalAveragePooling2D(name="ext_gap")(latent)
    x = layers.Dense(128, activation="relu", name="ext_dense")(x)
    x = layers.Dropout(0.3, name="ext_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="ext_pred")(x)
    
    classifier = Model(inputs=inputs, outputs=outputs, name="PretrainedEncoderClassifier")
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return classifier
