"""Autoencoder (AE) model architecture."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_ae_encoder(img_size=64, latent_dim=32, input_shape=None):
    """Build convolutional encoder for AE.
    
    Args:
        img_size: Size of input images
        latent_dim: Dimension of latent space
        input_shape: Input shape tuple
        
    Returns:
        keras.Model: Encoder model
    """
    if input_shape is None:
        input_shape = (img_size, img_size, 1)
    
    inputs = keras.Input(shape=input_shape, name='encoder_input')
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name='latent_code')(x)
    
    return keras.Model(inputs, latent, name='ae_encoder')


def build_ae_decoder(img_size=64, latent_dim=32):
    """Build convolutional decoder for AE.
    
    Args:
        img_size: Size of output images
        latent_dim: Dimension of latent space
        
    Returns:
        keras.Model: Decoder model
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(8 * 8 * 128, activation='relu')(latent_inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid', name='reconstruction')(x)
    
    return keras.Model(latent_inputs, outputs, name='ae_decoder')


def build_ae(img_size=64, latent_dim=32):
    """Build full Autoencoder: encoder + decoder.
    
    Args:
        img_size: Size of input images
        latent_dim: Dimension of latent space
        
    Returns:
        tuple: (full_model, encoder, decoder)
    """
    encoder = build_ae_encoder(img_size, latent_dim)
    decoder = build_ae_decoder(img_size, latent_dim)
    
    inputs = keras.Input(shape=(img_size, img_size, 1))
    latent = encoder(inputs)
    reconstructed = decoder(latent)
    
    return keras.Model(inputs, reconstructed, name='autoencoder'), encoder, decoder
