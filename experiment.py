import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from utils.data_loader import DataLoader
from utils.visualization import (
    visualize_samples, plot_training_losses, show_reconstructions,
    show_denoising, plot_latent_2d, generate_samples, interpolate_latent
)
from model.ae import build_ae
from model.vae import build_vae_encoder, build_vae_decoder, VAE
from train.train_ae import train_ae_for_all_classes
from train.train_vae import train_vae_for_all_classes


class MedicalMNISTExperiment:
    
    def __init__(self, dataset_path, img_size=64, batch_size=64, latent_dim=32, 
                 epochs=15, class_names=None, save_dir='results/models'):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.save_dir = save_dir
        
        self.data_loader = DataLoader(
            dataset_path, 
            img_size=img_size, 
            batch_size=batch_size,
            class_names=class_names
        )
        
        self.datasets = None
        self.ae_models = {}
        self.vae_models = {}
        self.ae_histories = {}
        self.vae_histories = {}
    
    def load_data(self):
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        self.datasets = self.data_loader.load_all_datasets()
        print(f"Loaded {len(self.datasets)} classes")
        return self.datasets
    
    def visualize_samples(self):
        if self.datasets is None:
            self.load_data()
        
        print("\n" + "="*60)
        print("VISUALIZING SAMPLES")
        print("="*60)
        visualize_samples(self.datasets, self.data_loader.class_names)
    
    def train_all_models(self):
        if self.datasets is None:
            self.load_data()
        
        print("\n" + "="*60)
        print("TRAINING AUTOENCODERS")
        print("="*60)
        self.ae_models, self.ae_histories = train_ae_for_all_classes(
            self.dataset_path,
            class_names=self.data_loader.class_names,
            epochs=self.epochs,
            batch_size=self.batch_size,
            img_size=self.img_size,
            save_dir=self.save_dir
        )
        
        print("\n" + "="*60)
        print("TRAINING VARIATIONAL AUTOENCODERS")
        print("="*60)
        self.vae_models, self.vae_histories = train_vae_for_all_classes(
            self.dataset_path,
            class_names=self.data_loader.class_names,
            epochs=self.epochs,
            batch_size=self.batch_size,
            img_size=self.img_size,
            latent_dim=self.latent_dim,
            save_dir=self.save_dir
        )
    
    def plot_losses(self):
        if not self.ae_histories or not self.vae_histories:
            raise ValueError("Models not trained yet. Call train_all_models() first.")
        
        print("\n" + "="*60)
        print("PLOTTING TRAINING LOSSES")
        print("="*60)
        plot_training_losses(self.ae_histories, self.vae_histories, 
                            self.data_loader.class_names)
    
    def analyze_reconstructions(self):
        if not self.ae_models or not self.vae_models:
            raise ValueError("Models not trained yet. Call train_all_models() first.")
        
        print("\n" + "="*60)
        print("ANALYZING RECONSTRUCTIONS")
        print("="*60)
        for cls in self.data_loader.class_names:
            show_reconstructions(cls, self.datasets, self.ae_models, self.vae_models)
    
    def analyze_denoising(self):
        if not self.ae_models or not self.vae_models:
            raise ValueError("Models not trained yet. Call train_all_models() first.")
        
        print("\n" + "="*60)
        print("ANALYZING DENOISING")
        print("="*60)
        for cls in self.data_loader.class_names:
            show_denoising(cls, self.datasets, self.ae_models, self.vae_models)
    
    def visualize_latent_space(self, method='pca'):
        if not self.ae_models or not self.vae_models:
            raise ValueError("Models not trained yet. Call train_all_models() first.")
        
        print("\n" + "="*60)
        print(f"VISUALIZING LATENT SPACE ({method.upper()})")
        print("="*60)
        for cls in self.data_loader.class_names:
            plot_latent_2d(cls, self.datasets, self.ae_models, self.vae_models, method=method)
    
    def generate_samples_vae(self, n_samples=10):
        if not self.vae_models:
            raise ValueError("VAE models not trained yet. Call train_all_models() first.")
        
        print("\n" + "="*60)
        print("GENERATING NEW SAMPLES FROM VAE")
        print("="*60)
        for cls in self.data_loader.class_names:
            generate_samples(cls, self.vae_models, self.latent_dim, n_samples)
    
    def interpolate_vae_latent(self, steps=10):
        if not self.vae_models:
            raise ValueError("VAE models not trained yet. Call train_all_models() first.")
        
        print("\n" + "="*60)
        print("INTERPOLATING IN VAE LATENT SPACE")
        print("="*60)
        for cls in self.data_loader.class_names:
            interpolate_latent(cls, self.vae_models, self.latent_dim, steps)
    
    def evaluate_mse(self, n_batches=3):
        if not self.ae_models or not self.vae_models:
            raise ValueError("Models not trained yet. Call train_all_models() first.")
        
        print("\n" + "="*60)
        print("QUANTITATIVE EVALUATION - MSE")
        print("="*60)
        print(f"{'Class':<15} {'AE MSE':>10} {'VAE MSE':>10}")
        print('-' * 38)
        
        for cls in self.data_loader.class_names:
            ds = self.datasets[cls]
            ae_m, _, _ = self.ae_models[cls]
            vae_m = self.vae_models[cls]
            
            ae_mses, vae_mses = [], []
            
            for batch, _ in ds.take(n_batches):
                ae_recon = ae_m.predict(batch, verbose=0)
                vae_recon = vae_m.predict(batch, verbose=0)
                
                ae_mses.append(np.mean((batch.numpy() - ae_recon) ** 2))
                vae_mses.append(np.mean((batch.numpy() - vae_recon) ** 2))
            
            ae_mse = np.mean(ae_mses)
            vae_mse = np.mean(vae_mses)
            
            print(f"{cls:<15} {ae_mse:>10.4f} {vae_mse:>10.4f}")
    
    def run_full_pipeline(self, visualize=True):
        print("\n" + "="*60)
        print("MEDICAL MNIST AUTOENCODER EXPERIMENT")
        print("="*60)
        
        self.load_data()
        
        if visualize:
            self.visualize_samples()
        
        self.train_all_models()
        
        if visualize:
            self.plot_losses()
            self.analyze_reconstructions()
            self.analyze_denoising()
            self.visualize_latent_space('pca')
            self.visualize_latent_space('tsne')
            self.generate_samples_vae()
            self.interpolate_vae_latent()
        
        self.evaluate_mse()
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE!")
        print("="*60)


if __name__ == '__main__':
    DATASET_PATH = "/content/drive/MyDrive/DSAI_490/archive (8).zip"
    
    experiment = MedicalMNISTExperiment(
        dataset_path=DATASET_PATH,
        img_size=64,
        batch_size=64,
        latent_dim=32,
        epochs=15,
        save_dir='results/models'
    )
    
    experiment.run_full_pipeline(visualize=True)
