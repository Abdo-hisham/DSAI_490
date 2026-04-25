"""Visualization utilities for analysis and results."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_samples(datasets, class_names, figsize=(12, 8)):
    """Visualize sample images from each class.
    
    Args:
        datasets: Dictionary of class_name -> dataset
        class_names: List of class names
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, (cls, ds) in enumerate(datasets.items()):
        batch, _ = next(iter(ds))
        axes[i].imshow(batch[0].numpy().squeeze(), cmap='gray')
        axes[i].set_title(cls, fontsize=12)
        axes[i].axis('off')

    plt.suptitle('Sample Images per Anatomical Region', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_training_losses(ae_histories, vae_histories, class_names, figsize=(14, 24)):
    """Plot training loss curves for AE and VAE.
    
    Args:
        ae_histories: Dictionary of class_name -> AE history
        vae_histories: Dictionary of class_name -> VAE history
        class_names: List of class names
        figsize: Figure size
    """
    fig, axes = plt.subplots(len(class_names), 2, figsize=figsize)

    for i, cls in enumerate(class_names):
        axes[i, 0].plot(ae_histories[cls].history['loss'], label='AE Loss', color='steelblue')
        axes[i, 0].set_title(f'{cls} — AE Reconstruction Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        vae_hist = vae_histories[cls].history
        axes[i, 1].plot(vae_hist['total_loss'], label='Total Loss', color='darkorange')
        axes[i, 1].plot(vae_hist['recon_loss'], label='Recon Loss', color='steelblue', linestyle='--')
        axes[i, 1].plot(vae_hist['kl_loss'], label='KL Loss', color='green', linestyle=':')
        axes[i, 1].set_title(f'{cls} — VAE Losses')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('Loss')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.suptitle('Training Loss Curves — AE vs VAE', fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.show()


def show_reconstructions(class_name, datasets, ae_models, vae_models, n=8):
    """Compare original vs AE vs VAE reconstructions.
    
    Args:
        class_name: Name of the class
        datasets: Dictionary of class_name -> dataset
        ae_models: Dictionary of class_name -> (model, encoder, decoder)
        vae_models: Dictionary of class_name -> model
        n: Number of samples to display
    """
    ds = datasets[class_name]
    batch, _ = next(iter(ds))
    imgs = batch[:n].numpy()

    ae_m, _, _ = ae_models[class_name]
    vae_m = vae_models[class_name]

    ae_recon = ae_m.predict(batch[:n], verbose=0)
    vae_recon = vae_m.predict(batch[:n], verbose=0)

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    titles = ['Original', 'AE Recon', 'VAE Recon']

    for row, (images, title) in enumerate(zip([imgs, ae_recon, vae_recon], titles)):
        for col in range(n):
            axes[row, col].imshow(images[col].squeeze(), cmap='gray')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=10, rotation=90, labelpad=40)

    plt.suptitle(f'Reconstructions — {class_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def add_gaussian_noise(images, noise_factor=0.3):
    """Add Gaussian noise to images.
    
    Args:
        images: Batch of images
        noise_factor: Noise magnitude
        
    Returns:
        Noisy images clipped to [0, 1]
    """
    noisy = images + noise_factor * tf.random.normal(shape=tf.shape(images))
    return tf.clip_by_value(noisy, 0.0, 1.0)


def show_denoising(class_name, datasets, ae_models, vae_models, n=6, noise_factor=0.3):
    """Demonstrate denoising capability of AE and VAE.
    
    Args:
        class_name: Name of the class
        datasets: Dictionary of class_name -> dataset
        ae_models: Dictionary of class_name -> (model, encoder, decoder)
        vae_models: Dictionary of class_name -> model
        n: Number of samples to display
        noise_factor: Noise magnitude
    """
    ds = datasets[class_name]
    batch, _ = next(iter(ds))
    imgs = batch[:n]
    noisy = add_gaussian_noise(imgs, noise_factor)

    ae_m, _, _ = ae_models[class_name]
    vae_m = vae_models[class_name]

    ae_denoised = ae_m.predict(noisy, verbose=0)
    vae_denoised = vae_m.predict(noisy, verbose=0)

    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8))
    rows = [imgs, noisy, ae_denoised, vae_denoised]
    labels = ['Original', 'Noisy Input', 'AE Denoised', 'VAE Denoised']

    for row_idx, (images, label) in enumerate(zip(rows, labels)):
        for col in range(n):
            axes[row_idx, col].imshow(
                images[col].numpy().squeeze() if hasattr(images[col], 'numpy') else images[col].squeeze(),
                cmap='gray'
            )
            axes[row_idx, col].axis('off')
            if col == 0:
                axes[row_idx, col].set_ylabel(label, fontsize=9, rotation=90, labelpad=40)

    plt.suptitle(f'Denoising — {class_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def get_latent_codes(class_name, datasets, ae_models, vae_models, model_type='ae', n_batches=5):
    """Extract latent codes from encoder.
    
    Args:
        class_name: Name of the class
        datasets: Dictionary of class_name -> dataset
        ae_models: Dictionary of class_name -> (model, encoder, decoder)
        vae_models: Dictionary of class_name -> model
        model_type: 'ae' or 'vae'
        n_batches: Number of batches to process
        
    Returns:
        Array of latent codes
    """
    ds = datasets[class_name]
    codes = []

    for batch, _ in ds.take(n_batches):
        if model_type == 'ae':
            _, ae_enc, _ = ae_models[class_name]
            z = ae_enc.predict(batch, verbose=0)
        else:
            vae_m = vae_models[class_name]
            z_mean, _, _ = vae_m.encoder.predict(batch, verbose=0)
            z = z_mean
        codes.append(z)

    return np.concatenate(codes, axis=0)


def plot_latent_2d(class_name, datasets, ae_models, vae_models, method='pca', figsize=(14, 5)):
    """Visualize 2D latent space using PCA or t-SNE.
    
    Args:
        class_name: Name of the class
        datasets: Dictionary of class_name -> dataset
        ae_models: Dictionary of class_name -> (model, encoder, decoder)
        vae_models: Dictionary of class_name -> model
        method: 'pca' or 'tsne'
        figsize: Figure size
    """
    ae_codes = get_latent_codes(class_name, datasets, ae_models, vae_models, model_type='ae')
    vae_codes = get_latent_codes(class_name, datasets, ae_models, vae_models, model_type='vae')

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, codes, title in zip(axes, [ae_codes, vae_codes], ['AE Latent Space', 'VAE Latent Space']):
        if method == 'pca':
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(codes)
            xlabel, ylabel = 'PC1', 'PC2'
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = reducer.fit_transform(codes)
            xlabel, ylabel = 't-SNE 1', 't-SNE 2'

        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5, s=10,
                           c=range(len(embedded)), cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Sample index')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Latent Space — {class_name} ({method.upper()})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def generate_samples(class_name, vae_models, latent_dim=32, n=10):
    """Generate new samples from VAE.
    
    Args:
        class_name: Name of the class
        vae_models: Dictionary of class_name -> model
        latent_dim: Dimension of latent space
        n: Number of samples to generate
    """
    vae_m = vae_models[class_name]

    z_samples = np.random.normal(size=(n, latent_dim)).astype(np.float32)
    generated = vae_m.decoder.predict(z_samples, verbose=0)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
    for i in range(n):
        axes[i].imshow(generated[i].squeeze(), cmap='gray')
        axes[i].axis('off')

    plt.suptitle(f'VAE Generated Samples — {class_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def interpolate_latent(class_name, vae_models, latent_dim=32, steps=10):
    """Interpolate between two points in VAE latent space.
    
    Args:
        class_name: Name of the class
        vae_models: Dictionary of class_name -> model
        latent_dim: Dimension of latent space
        steps: Number of interpolation steps
    """
    vae_m = vae_models[class_name]

    z_a = np.random.normal(size=(1, latent_dim)).astype(np.float32)
    z_b = np.random.normal(size=(1, latent_dim)).astype(np.float32)

    alphas = np.linspace(0, 1, steps)
    interpolated = np.array([(1 - a) * z_a + a * z_b for a in alphas]).squeeze()

    decoded = vae_m.decoder.predict(interpolated, verbose=0)

    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 3))
    for i in range(steps):
        axes[i].imshow(decoded[i].squeeze(), cmap='gray')
        axes[i].set_title(f'α={alphas[i]:.1f}', fontsize=8)
        axes[i].axis('off')

    plt.suptitle(f'VAE Latent Interpolation — {class_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
