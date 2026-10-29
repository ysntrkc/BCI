# *MIST HER2 - Baseline vs DWT Comparison*

## *Experiment 1 - Baseline Model with Batch Size 20*

### Train
```bash
nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
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
  > 'checkpoints/comparison_baseline/training_output_baseline_mist_20260107_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
  --name comparison_baseline \
  --run_name run_20260107_205347_MIST/HER2 \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path ./results/comparison_baseline/run_20260107_205347_MIST/HER2
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  1000
------------------------------------------------------------
Average PSNR:      15.322830 dB
Average SSIM:      0.252960
FID Score:         173.051529
KID Score:         0.130980
JSD Score:         0.120575
============================================================
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 15.322830  |
| Average SSIM  ↑    | 0.252960  |
| FID Score     ↓    | 173.051529 |
| KID Score (x1000) ↓ | 130.980  |
| JSD Score         ↓ | 0.120575  |



## *Experiment 2 - Baseline Model with Batch Size 16*

### Train
```bash
nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
  --name comparison_baseline \
  --pattern L1_L2_L3 \
  --display_id -1 \
  --batch_size 16 \
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
  > 'checkpoints/comparison_baseline/training_output_baseline_mist_20260108_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
  --name comparison_baseline \
  --run_name run_20260108_003633_MIST_HER2 \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path ./results/comparison_baseline/run_20260108_003633_MIST_HER2
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  1000
------------------------------------------------------------
Average PSNR:      15.241194 dB
Average SSIM:      0.249908
FID Score:         161.157814
KID Score:         0.116079
JSD Score:         0.109859
============================================================
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 15.241194  |
| Average SSIM  ↑    | 0.249908  |
| FID Score     ↓    | 161.157814 |
| KID Score (x1000) ↓ | 116.079  |
| JSD Score         ↓ | 0.109859  |



## *Experiment 3 - DWT Model with Batch Size 20*

### Train
```bash
nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
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
  > 'checkpoints/comparison_dwt/training_output_dwt_mist_20260108_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
  --name comparison_dwt \
  --run_name run_20260108_075951_MIST_HER2 \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
```

### Evaluate
```bash
python evaluate.py \
  --result_path ./results/comparison_dwt/run_20260108_075951_MIST_HER2
```

<!-- 
============================================================
                    EVALUATION SUMMARY
============================================================
Number of images:  1000
------------------------------------------------------------
Average PSNR:      15.297673 dB
Average SSIM:      0.259146
FID Score:         136.946822
KID Score:         0.091966
JSD Score:         0.117865
============================================================
 -->

### Results
| Metric | Value |
|--------|-------|
| Average PSNR  ↑    | 15.297673  |
| Average SSIM  ↑    | 0.259146  |
| FID Score     ↓    | 136.946822 |
| KID Score (x1000) ↓ | 91.966  |
| JSD Score         ↓ | 0.117865  |


# Overall Summary of Results

| Experiment | Model                             | Average PSNR  ↑ | Average SSIM  ↑ | FID Score ↓ | KID Score (x1000) ↓ | JSD Score ↓ |
|------------|-----------------------------------|-----------------|-----------------|-------------|---------------------|--------------|
| 1          | Baseline Model (Batch Size 20)    | 15.3228        | 0.2529         | 173.0515    | 130.980             | 0.1206       |
| 2          | Baseline Model (Batch Size 16)    | 15.2412        | 0.2499         | 161.1578    | 116.079             | 0.1099       |
| 3          | DWT Model (Batch Size 20)         | 15.2977        | 0.2591         | 136.9468    | 91.966              | 0.1179       |