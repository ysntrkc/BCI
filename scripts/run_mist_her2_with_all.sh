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


# Test baseline
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
  --name comparison_baseline \
  --run_name run_20260108_003633_MIST_HER2 \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval


# Evaluate baseline results
python evaluate.py \
  --result_path ./results/comparison_baseline/run_20260108_003633_MIST_HER2


The average psnr is 15.241194163197815
The average ssim is 0.24990762405411435




# Train with DWT
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


# Test DWT
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
  --name comparison_dwt \
  --run_name run_20260107_095053_BCI \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval


# Evaluate DWT results
python evaluate.py \
  --result_path ./results/comparison_dwt/run_20260107_095053_BCI


The average psnr is 22.549359913171948
The average ssim is 0.5763130591454685