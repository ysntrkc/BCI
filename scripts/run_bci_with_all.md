# BCI - Baseline vs DWT Comparison

## Experiment 1 - Baseline Model

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

### Results
| Metric | Value |
|--------|-------|
| Average PSNR | 21.885746928999986 |
| Average SSIM | 0.5349153675040248 |



## Experiment 2 - DWT Model

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

### Results
| Metric | Value |
|--------|-------|
| Average PSNR | 22.549359913171948 |
| Average SSIM | 0.5763130591454685 |


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

### Results
| Metric | Value |
|--------|-------|
| Average PSNR | 19.387844116353552 |
| Average SSIM | 0.4471386262106031 |


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

### Results
| Metric | Value |
|--------|-------|
| Average PSNR | 20.219152241068073 |
| Average SSIM | 0.43168893303754113 |
