import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vae import build_vae_encoder, build_vae_decoder, VAE
from utils.data_loader import DataLoader


def train_vae_for_all_classes(dataset_path, class_names=None, epochs=15, batch_size=64,
                              img_size=64, latent_dim=32, beta=1.0, save_dir='results/models'):
    os.makedirs(save_dir, exist_ok=True)
    
    data_loader = DataLoader(dataset_path, img_size=img_size, batch_size=batch_size,
                            class_names=class_names)
    datasets = data_loader.load_all_datasets()
    
    vae_models = {}
    vae_histories = {}
    
    for cls in data_loader.class_names:
        print(f"\n{'='*60}")
        print(f"Training VAE for: {cls}")
        print(f"{'='*60}")
        
        ds = datasets[cls]
        
        vae_enc = build_vae_encoder(img_size=img_size, latent_dim=latent_dim)
        vae_dec = build_vae_decoder(img_size=img_size, latent_dim=latent_dim)
        vae_m = VAE(vae_enc, vae_dec, beta=beta)
        vae_m.compile(optimizer='adam')
        
        vae_hist = vae_m.fit(
            ds,
            epochs=epochs,
            verbose=1
        )
        
        vae_models[cls] = vae_m
        vae_histories[cls] = vae_hist
        
        vae_m.save(os.path.join(save_dir, f'vae_{cls}'))
        print(f"Saved: vae_{cls}")
    
    print(f"\n{'='*60}")
    print("All VAE models trained and saved!")
    print(f"{'='*60}")
    
    return vae_models, vae_histories


if __name__ == '__main__':
    # Example usage
    DATASET_PATH = "/content/drive/MyDrive/DSAI_490/archive (8).zip"
    train_vae_for_all_classes(DATASET_PATH, epochs=15, batch_size=64)
