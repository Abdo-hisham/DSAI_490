# Medical MNIST Autoencoder Analysis

Analysis and comparison of Autoencoder (AE) and Variational Autoencoder (VAE) models trained on Medical MNIST dataset across 6 anatomical regions.

## Project Structure

```
├── model/
│   ├── ae.py              # Autoencoder architecture
│   └── vae.py             # Variational Autoencoder architecture
├── train/
│   ├── train_ae.py        # AE training script
│   └── train_vae.py       # VAE training script
├── utils/
│   ├── data_loader.py     # Dataset loading and preprocessing
│   └── visualization.py   # Visualization functions
├── results/
│   ├── models/            # Trained model checkpoints
│   └── s.txt              # Results summary
├── notebooks/
│   └── Experiment.ipynb   # Interactive experiment notebook
├── experiment.py          # Main experiment runner
├── requirements.txt       # Python dependencies
└── README.md
```

## Anatomical Regions

The dataset contains 6 classes of medical images:
- **AbdomenCT**: Abdominal CT scans
- **BreastMRI**: Breast MRI scans  
- **ChestCT**: Chest CT scans
- **CXR**: Chest X-Ray images
- **Hand**: Hand X-Ray images
- **HeadCT**: Head CT scans

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

```python
from experiment import MedicalMNISTExperiment

experiment = MedicalMNISTExperiment(
    dataset_path="/path/to/dataset.zip",
    img_size=64,
    batch_size=64,
    latent_dim=32,
    epochs=15
)

experiment.run_full_pipeline(visualize=True)
```

### Individual Components

#### Train Models

```python
experiment.train_all_models()
```

#### Evaluate Models

```python
experiment.evaluate_mse()
experiment.plot_losses()
```

#### Generate Visualizations

```python
experiment.visualize_samples()
experiment.analyze_reconstructions()
experiment.analyze_denoising()
experiment.visualize_latent_space(method='pca')
experiment.generate_samples_vae()
experiment.interpolate_vae_latent()
```

## Model Architectures

### Autoencoder (AE)

**Encoder:**
- Conv2D(32, 3, strides=2) + ReLU
- Conv2D(64, 3, strides=2) + ReLU
- Conv2D(128, 3, strides=2) + ReLU
- Flatten + Dense(latent_dim)

**Decoder:**
- Dense(8×8×128) + ReLU
- Reshape to (8, 8, 128)
- Conv2DTranspose(128, 3, strides=2) + ReLU
- Conv2DTranspose(64, 3, strides=2) + ReLU
- Conv2DTranspose(32, 3, strides=2) + ReLU
- Conv2DTranspose(1, 3, sigmoid activation)

### Variational Autoencoder (VAE)

Same architecture as AE with:
- Sampling layer implementing reparameterization trick
- KL divergence loss component
- Custom train_step for loss computation

## Training Configuration

- **Latent Dimension**: 32
- **Image Size**: 64×64
- **Batch Size**: 64
- **Epochs**: 15
- **Optimizer**: Adam
- **Loss (AE)**: Binary Crossentropy
- **Loss (VAE)**: Reconstruction Loss + KL Divergence

## Experiments

### 1. Reconstruction Quality
Compares original vs AE vs VAE reconstructions for each class.

### 2. Denoising Capability
Tests robustness to Gaussian noise (factor: 0.3).

### 3. Latent Space Analysis
Visualizes latent space using PCA and t-SNE dimensionality reduction.

### 4. Sample Generation
Generates new synthetic medical images from VAE latent space.

### 5. Latent Space Interpolation
Demonstrates smooth transitions between generated samples.

### 6. Quantitative Evaluation
Computes Mean Squared Error (MSE) for reconstruction quality.

## Key Features

✓ Modular architecture for easy experimentation
✓ Support for multiple anatomical regions
✓ Comprehensive visualization suite
✓ Quantitative and qualitative evaluation
✓ Model checkpointing and loading
✓ Colab-compatible data loading

## Results Directory

Models and evaluation results are saved to `results/`:
- Model checkpoints: `results/models/ae_*.h5`, `results/models/vae_*`
- Summary statistics: `results/s.txt`

## References

- Medical MNIST Dataset: https://medmnist.com/
- Autoencoders: Kingma & Welling (2013)
- VAE: Auto-Encoding Variational Bayes

## Author

DSAI 490 - Medical Image Analysis Course Project
