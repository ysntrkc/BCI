# BCI - Baseline vs DWT Comparison

## Experiment 1 - Pyramid Pix2Pix (2 layer) Baseline Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI/combined \
  --name comparison_baseline \
  --pattern L1_L2_L3 \
  --display_id -1 \
  --batch_size 20 \
  --num_threads 8 \
  --load_size 320 \
  --crop_size 256 \
  --preprocess crop \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  > 'checkpoints/comparison_baseline/training_output_baseline_bci_20260107_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI/combined \
  --name comparison_baseline \
  --run_name run_20260107_040153_BCI \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path ./results/comparison_baseline/run_20260107_040153_BCI
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  977
------------------------------------------------------------
Average PSNR:      21.885747 dB
Average SSIM:      0.534915
FID Score:         149.346252
KID Score:         0.112267
JSD Score:         0.192756
============================================================
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 21.885747  |
| Average SSIM  ↑    | 0.534915  |
| FID Score     ↓    | 149.346252 |
| KID Score (x1000) ↓ | 112.267  |
| JSD Score         ↓ | 0.192756  |


## Experiment 2 - Pyramid Pix2Pix (2 layer) DWT Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI/combined \
  --name comparison_dwt \
  --pattern L1_L2_L3_dwt \
  --display_id -1 \
  --batch_size 20 \
  --num_threads 8 \
  --load_size 320 \
  --crop_size 256 \
  --preprocess crop \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  > 'checkpoints/comparison_dwt/training_output_dwt_bci_20260107_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI/combined \
  --name comparison_dwt \
  --run_name run_20260107_095053_BCI \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path ./results/comparison_dwt/run_20260107_095053_BCI
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  977
------------------------------------------------------------
Average PSNR:      22.549360 dB
Average SSIM:      0.576313
FID Score:         118.832649
KID Score:         0.074003
JSD Score:         0.197845
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 22.549360  |
| Average SSIM  ↑    | 0.576313  |
| FID Score     ↓    | 118.832649 |
| KID Score (x1000) ↓ | 74.003  |
| JSD Score         ↓ | 0.197845  |

## Experiment 3 - CycleGAN Baseline Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name cyclegan_baseline \
  --model cycle_gan \
  --display_id -1 \
  --batch_size 4 \
  --num_threads 8 \
  --load_size 320 \
  --crop_size 256 \
  --preprocess crop \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  > '/mnt/d/thesis/checkpoints/cyclegan_baseline/training_output_cyclegan_baseline_bci_20260110_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name cyclegan_baseline \
  --model cycle_gan \
  --run_name run_20260110_024846_BCI \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path /mnt/d/thesis/results/cyclegan_baseline/run_20260110_024846_BCI
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  977
------------------------------------------------------------
Average PSNR:      19.387844 dB
Average SSIM:      0.447139
FID Score:         85.050197
KID Score:         0.042423
JSD Score:         0.330946
============================================================
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 19.387844  |
| Average SSIM  ↑    | 0.447139  |
| FID Score     ↓    | 85.050197 |
| KID Score (x1000) ↓ | 42.423  |
| JSD Score         ↓ | 0.330946  |

## Experiment 4 - CycleGAN DWT Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name cyclegan_dwt \
  --model cycle_gan \
  --use_dwt \
  --weight_dwt_ll 25.0 \
  --weight_dwt_detail 25.0 \
  --display_id -1 \
  --batch_size 4 \
  --num_threads 8 \
  --load_size 320 \
  --crop_size 256 \
  --preprocess crop \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  > '/mnt/d/thesis/checkpoints/training_output_cyclegan_dwt_bci_20260109_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name cyclegan_dwt \
  --model cycle_gan \
  --run_name run_20260109_124343_BCI \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path /mnt/d/thesis/results/cyclegan_dwt/run_20260109_124343_BCI
```

<!--
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  977
------------------------------------------------------------
Average PSNR:      20.219152 dB
Average SSIM:      0.431689
FID Score:         74.243779
KID Score:         0.039676
JSD Score:         0.241232
============================================================
-->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 20.219152  |
| Average SSIM  ↑    | 0.431689  |
| FID Score     ↓    | 74.243779 |
| KID Score (x1000) ↓ | 39.676  |
| JSD Score         ↓ | 0.241232  |

## Experiment 5 - Pix2Pix Baseline Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name pix2pix_baseline \
  --model pix2pix \
  --pattern L1 \
  --display_id -1 \
  --batch_size 20 \
  --num_threads 8 \
  --load_size 320 \
  --crop_size 256 \
  --preprocess crop \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  > '/mnt/d/thesis/checkpoints/pix2pix_baseline/training_output_pix2pix_baseline_bci_20260113_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name pix2pix_baseline \
  --model pix2pix \
  --run_name run_20260113_030327_BCI \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path /mnt/d/thesis/results/pix2pix_baseline/run_20260113_030327_BCI
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  977
------------------------------------------------------------
Average PSNR:      21.393719 dB
Average SSIM:      0.528243
FID Score:         177.347372
KID Score:         0.144712
JSD Score:         0.227408
============================================================
 -->


### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 21.393719  |
| Average SSIM  ↑    | 0.528243  |
| FID Score     ↓    | 177.355687 |
| KID Score (x1000) ↓ | 144.712  |
| JSD Score         ↓ | 0.227408  |


## Experiment 6 - Pix2Pix DWT Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name pix2pix_dwt \
  --model pix2pix \
  --pattern L1_dwt \
  --display_id -1 \
  --batch_size 20 \
  --num_threads 8 \
  --load_size 320 \
  --crop_size 256 \
  --preprocess crop \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  > '/mnt/d/thesis/checkpoints/pix2pix_dwt/training_output_pix2pix_dwt_bci_20260113_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name pix2pix_dwt \
  --model pix2pix \
  --run_name run_20260113_093136_BCI \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path /mnt/d/thesis/results/pix2pix_dwt/run_20260113_093136_BCI
```
<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  977
------------------------------------------------------------
Average PSNR:      21.955590 dB
Average SSIM:      0.569812
FID Score:         120.666121
KID Score:         0.077398
JSD Score:         0.200150
============================================================
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 21.955590  |
| Average SSIM  ↑    | 0.569812  |
| FID Score     ↓    | 120.666121 |
| KID Score (x1000) ↓ | 77.398  |
| JSD Score         ↓ | 0.200150  |


## Experiment 7 - Pyramid Pix2Pix (2 layer) Baseline Model with original image size 1024x1024 and crop size 512x512

### Train
```bash
nohup python train.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name comparison_baseline_1024 \
  --pattern L1_L2_L3 \
  --display_id -1 \
  --batch_size 4 \
  --num_threads 8 \
  --load_size 1024 \
  --crop_size 512 \
  --preprocess crop \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 50 \
  --n_epochs_decay 50 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  > '/mnt/d/thesis/checkpoints/comparison_baseline_1024/training_output_baseline_bci_20260118_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/d/thesis/datasets/BCI/combined \
  --name comparison_baseline_1024 \
  --run_name run_20260118_170529_BCI \
  --preprocess crop \
  --load_size 1024 \
  --crop_size 512 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path /mnt/d/thesis/results/comparison_baseline_1024/run_20260118_170529_BCI
```


# Overall Summary of Results

| Experiment | Model Type                         | Average PSNR ↑ | Average SSIM ↑ | FID Score ↓ | KID Score (x1000) ↓ | JSD Score ↓  |
|------------|------------------------------------|----------------|----------------|-------------|---------------------|--------------|
| 1          | Pyramid Pix2Pix (2 layer)          | 21.8857        | 0.5349         | 149.3463    | 112.267             | 0.1928       |
| 2          | Pyramid Pix2Pix (2 layer) DWT      | 22.5494        | 0.5763         | 118.8326    | 74.003              | 0.1978       |
| 3          | CycleGAN Baseline                  | 19.3878        | 0.4471         | 85.0502     | 42.423              | 0.3309       |
| 4          | CycleGAN DWT                       | 20.2192        | 0.4317         | 74.2438     | 39.676              | 0.2412       |
| 5          | Pix2Pix Baseline                   | 21.3937        | 0.5282         | 177.3474    | 144.712             | 0.2274       |
| 6          | Pix2Pix DWT                        | 21.9556        | 0.5698         | 120.6661    | 77.398              | 0.2002       | 

## Notes

**DWT Models:** All experiments labeled with "DWT" utilize Haar Wavelet Transform loss in addition to standard loss functions for enhanced frequency-domain optimization. We expect these models to perform better in terms of perceptual quality and certain quantitative metrics.
