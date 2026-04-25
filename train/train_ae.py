"""Train Autoencoder models for each anatomical region."""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from model.ae import build_ae
from utils.data_loader import DataLoader


def train_ae_for_all_classes(dataset_path, class_names=None, epochs=15, batch_size=64, 
                             img_size=64, save_dir='results/models'):
    """Train Autoencoder for each anatomical region.
    
    Args:
        dataset_path: Path to dataset or zip file
        class_names: List of class names
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        save_dir: Directory to save models
        
    Returns:
        dict: Dictionary of class_name -> (model, encoder, decoder, history)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    data_loader = DataLoader(dataset_path, img_size=img_size, batch_size=batch_size,
                            class_names=class_names)
    datasets = data_loader.load_all_datasets()
    
    ae_models = {}
    ae_histories = {}
    
    for cls in data_loader.class_names:
        print(f"\n{'='*60}")
        print(f"Training AE for: {cls}")
        print(f"{'='*60}")
        
        ds = datasets[cls]
        
        ae_m, ae_enc, ae_dec = build_ae(img_size=img_size)
        ae_m.compile(optimizer='adam', loss='binary_crossentropy')
        
        ae_hist = ae_m.fit(
            ds,
            epochs=epochs,
            verbose=1
        )
        
        ae_models[cls] = (ae_m, ae_enc, ae_dec)
        ae_histories[cls] = ae_hist
        
        ae_m.save(os.path.join(save_dir, f'ae_{cls}.h5'))
        print(f"Saved: ae_{cls}.h5")
    
    print(f"\n{'='*60}")
    print("All AE models trained and saved!")
    print(f"{'='*60}")
    
    return ae_models, ae_histories


if __name__ == '__main__':
    # Example usage
    DATASET_PATH = "/content/drive/MyDrive/DSAI_490/archive (8).zip"
    train_ae_for_all_classes(DATASET_PATH, epochs=15, batch_size=64)
