"""Variational Autoencoder (VAE) model architecture."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Custom sampling layer for VAE."""
    
    def call(self, inputs):
        """Apply reparameterization trick.
        
        Args:
            inputs: tuple of (z_mean, z_log_var)
            
        Returns:
            Sampled latent vector
        """
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_vae_encoder(img_size=64, latent_dim=32):
    """Build VAE encoder with reparameterization.
    
    Args:
        img_size: Size of input images
        latent_dim: Dimension of latent space
        
    Returns:
        keras.Model: Encoder returning [z_mean, z_log_var, z]
    """
    inputs = keras.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    return keras.Model(inputs, [z_mean, z_log_var, z], name='vae_encoder')


def build_vae_decoder(img_size=64, latent_dim=32):
    """Build VAE decoder.
    
    Args:
        img_size: Size of output images
        latent_dim: Dimension of latent space
        
    Returns:
        keras.Model: Decoder model
    """
    inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 128, activation='relu')(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
    
    return keras.Model(inputs, x, name='vae_decoder')


class VAE(keras.Model):
    """Variational Autoencoder model with custom training loop."""
    
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        """Initialize VAE.
        
        Args:
            encoder: VAE encoder model
            decoder: VAE decoder model
            beta: KL divergence weight
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        """Return tracked metrics."""
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        """Custom training step.
        
        Args:
            data: Batch of training data
            
        Returns:
            dict: Loss metrics
        """
        images, _ = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(images, training=True)
            reconstructed = self.decoder(z, training=True)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(images, reconstructed), axis=(1, 2))
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        """Forward pass through VAE.
        
        Args:
            inputs: Input images
            
        Returns:
            Reconstructed images
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)
