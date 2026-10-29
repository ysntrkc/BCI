# Comprehensive Results Summary

This document consolidates all experimental results across different datasets and model architectures.

---

## 1. BCI Dataset Results

### Model Comparison on BCI Dataset

| Experiment | Model Type                         | Average PSNR ↑ | Average SSIM ↑ | FID Score ↓ | KID Score (x1000) ↓ |
|------------|------------------------------------|----------------|----------------|-------------|---------------------|
| 3          | CycleGAN                           | 19.3878        | 0.4471         | 85.0502     | 42.423              |
| 4          | CycleGAN DWT                       | 20.2192        | 0.4317         | 74.2438     | 39.676              |
| 5          | Pix2Pix                            | 21.3937        | 0.5282         | 177.3474    | 144.712             |
| 6          | Pix2Pix DWT                        | 21.9556        | 0.5698         | 120.6661    | 77.398              |
| 1          | Pyramid Pix2Pix (2 layer)          | 21.8857        | 0.5349         | 149.3463    | 112.267             |
| 2          | Pyramid Pix2Pix (2 layer) DWT      | 22.5494        | 0.5763         | 118.8326    | 74.003              |

### Key Findings (BCI Dataset)
- **Best PSNR**: Pyramid Pix2Pix DWT (22.5494 dB)
- **Best SSIM**: Pyramid Pix2Pix DWT (0.5763)
- **Best FID**: CycleGAN DWT (74.2438)
- **Best KID**: CycleGAN DWT (39.676)

### Notes
- **Number of test images**: 977
- **Image specifications**: Load size 320, Crop size 256
- **Training parameters**: 50 epochs + 50 decay epochs, LR 0.0002
- **DWT Models**: All experiments labeled with "DWT" utilize Haar Wavelet Transform loss in addition to standard loss functions for enhanced frequency-domain optimization

---

## 2. Pyramid Pix2Pix - Three Dataset Results

### Comprehensive Pyramid Pix2Pix Comparison Across Datasets

| Dataset       | Model                             | Test Images | Average PSNR ↑ | Average SSIM ↑ | FID Score ↓ | KID Score (x1000) ↓ |
|---------------|-----------------------------------|-------------|----------------|----------------|-------------|---------------------|
| BCI           | Pyramid Pix2Pix (2 layer)         | 977         | 21.8857        | 0.5349         | 149.3463    | 112.267             |
| BCI           | Pyramid Pix2Pix (2 layer) DWT     | 977         | 22.5494        | 0.5763         | 118.8326    | 74.003              |
| MIST HER2     | Baseline Model                    | 1000        | 15.3228        | 0.2529         | 173.0515    | 130.980             |
| MIST HER2     | DWT Model                         | 1000        | 15.2977        | 0.2591         | 136.9468    | 91.966              |
| HER2match     | Baseline                          | 5980        | 19.7859        | 0.4451         | 66.1814     | 62.622              |
| HER2match     | DWT                               | 5980        | 19.8542        | 0.4477         | 59.1161     | 54.742              |

---

## Cross-Dataset Analysis

### DWT Improvement Analysis

#### BCI Dataset (Pyramid Pix2Pix)
- PSNR improvement: +3.04% (21.8857 → 22.5494)
- SSIM improvement: +7.74% (0.5349 → 0.5763)
- FID improvement: -20.48% (149.3463 → 118.8326)
- KID improvement: -34.04% (112.267 → 74.003)

#### MIST HER2 Dataset
- PSNR improvement: -0.16% (15.3228 → 15.2977) - comparing batch 20 models
- SSIM improvement: +2.45% (0.2529 → 0.2591)
- FID improvement: -20.84% (173.0515 → 136.9468)
- KID improvement: -29.77% (130.980 → 91.966)

#### HER2match Dataset
- PSNR improvement: +0.35% (19.7859 → 19.8542)
- SSIM improvement: +0.58% (0.4451 → 0.4477)
- FID improvement: -10.68% (66.1814 → 59.1161)
- KID improvement: -12.58% (62.622 → 54.742)

### General Observations

1. **DWT Enhancement**: DWT models consistently show improvements in perceptual metrics (FID, KID) across all datasets
2. **Dataset Difficulty**: MIST HER2 appears to be the most challenging dataset with the lowest PSNR and SSIM scores
3. **Best Performance**: BCI dataset yields the best overall metrics, likely due to optimal data characteristics
4. **Model Architecture**: Pyramid Pix2Pix DWT demonstrates superior performance compared to standard Pix2Pix and CycleGAN architectures on the BCI dataset

---

## Metric Interpretation Guide

- **PSNR (Peak Signal-to-Noise Ratio)** ↑: Higher is better. Measures pixel-level accuracy (typical range: 15-25 dB)
- **SSIM (Structural Similarity Index)** ↑: Higher is better. Measures structural similarity (range: 0-1)
- **FID (Fréchet Inception Distance)** ↓: Lower is better. Measures distribution similarity in feature space
- **KID (Kernel Inception Distance)** ↓: Lower is better. Alternative to FID with better small-sample properties

---
