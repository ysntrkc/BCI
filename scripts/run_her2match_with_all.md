# HER2match - Baseline vs DWT Comparison

## Experiment 1 - Baseline Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/HER2match/combined \
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
  > 'checkpoints/comparison_baseline/training_output_baseline_her2match_20260108_001.txt' 2>&1 &
```

### Test
```bash
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/HER2match/combined \
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

### Results
| Metric | Value |
|--------|-------|
| Average PSNR | 19.785883393511664 |
| Average SSIM | 0.4451300986370742 |



## Experiment 2 - DWT Model

### Train
```bash
nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/HER2match/combined \
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
  --dataroot /mnt/c/users/gaming/datasets/HER2match/combined \
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

### Results
| Metric | Value |
|--------|-------|
| Average PSNR | 19.854245689031934 |
| Average SSIM | 0.44767109127532906 |
