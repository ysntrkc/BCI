# HER2match - Baseline vs DWT Comparison

## Experiment 1 - Baseline Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/d/thesis/datasets/HER2match/combined \
  --name pyramidpix2pix \
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
  > 'checkpoints/pyramidpix2pix/training_output_baseline_her2match_20260108_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/d/thesis/datasets/HER2match/combined \
  --name pyramidpix2pix \
  --run_name run_004_20251231_HER2match \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path ./results/pyramidpix2pix/run_004_20251231_HER2match
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  5980
------------------------------------------------------------
Average PSNR:      19.785883 dB
Average SSIM:      0.445130
FID Score:         66.181411
KID Score:         0.062622
JSD Score:         0.060695
============================================================
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 19.785883  |
| Average SSIM  ↑    | 0.445130  |
| FID Score     ↓    | 66.181411 |
| KID Score (x1000) ↓ | 62.622  |
| JSD Score         ↓ | 0.060695  |



## Experiment 2 - DWT Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/d/thesis/datasets/HER2match/combined \
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
  > 'checkpoints/comparison_dwt/training_output_dwt_her2match_20260108_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/d/thesis/datasets/HER2match/combined \
  --name comparison_dwt \
  --run_name run_20260108_114345_HER2match \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path ./results/comparison_dwt/run_20260108_114345_HER2match
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  5980
------------------------------------------------------------
Average PSNR:      19.854246 dB
Average SSIM:      0.447671
FID Score:         59.116100
KID Score:         0.054742
JSD Score:         0.065014
============================================================
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 19.854246  |
| Average SSIM  ↑    | 0.447671  |
| FID Score     ↓    | 59.116100 |
| KID Score (x1000) ↓ | 54.742  |
| JSD Score         ↓ | 0.065014  |


# Overall Summary of Results
| Experiment | Model                             | Average PSNR  ↑ | Average SSIM  ↑ | FID Score ↓ | KID Score (x1000) ↓ | JSD Score ↓ |
|------------|-----------------------------------|-----------------|-----------------|-------------|---------------------|--------------|
| 1          | Baseline                          | 19.785883       | 0.445130        | 66.181411   | 62.622              | 0.060695     |
| 2          | DWT                               | 19.854246       | 0.447671        | 59.116100   | 54.742              | 0.065014     |