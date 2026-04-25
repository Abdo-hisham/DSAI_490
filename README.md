# Medical Image Autoencoders

A hands-on project comparing different types of neural networks that learn to compress and reconstruct medical images.

## What's This About?

We train two types of smart neural networks (Autoencoders) on medical images from 6 different anatomical regions. The networks learn to:
- **Compress** images into a smaller representation
- **Reconstruct** the original image from that compressed version
- **Denoise** noisy/blurry images

## The Medical Images

We work with images from:
- Abdominal CT scans
- Breast MRI scans
- Chest CT scans
- Chest X-Rays
- Hand X-Rays
- Head CT scans

## Project Structure

```
DSAI_490/
├── model/               # Network architectures
│   ├── ae.py           # Standard Autoencoder
│   └── vae.py          # Variational Autoencoder
├── train/              # Training scripts
│   ├── train_ae.py
│   └── train_vae.py
├── utils/              # Helper functions
│   ├── data_loader.py
│   └── visualization.py
├── notebooks/
│   └── Abdo_490.ipynb  # Interactive notebook with all experiments
└── requirements.txt    # Python packages needed
```

## Two Types of Networks

### 1. Autoencoder (AE)
A straightforward network that compresses and reconstructs images.

### 2. Variational Autoencoder (VAE)
A smarter version that also learns the structure of the data, allowing us to:
- Generate completely new images
- Smoothly blend between different images

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open `notebooks/Abdo_490.ipynb` and run the cells to see the full experiment.

### 3. What You'll See
- Original vs reconstructed medical images
- How well networks remove noise from images
- 2D visualizations of what the networks learned
- Generated new synthetic medical images
- Smooth transitions between different images

## Key Results

The notebook shows:
✓ Reconstruction quality comparison  
✓ Denoising capability  
✓ Latent space analysis  
✓ Generated sample images  
✓ Performance metrics (MSE)

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy, Matplotlib
- Scikit-learn
- Access to Medical MNIST dataset

## Learn More

This project demonstrates:
- How neural networks compress information
- The difference between deterministic and probabilistic models
- Generative modeling techniques
- Medical image analysis basics

Enjoy exploring! 🎉
