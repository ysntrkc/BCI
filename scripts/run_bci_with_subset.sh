python train.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI/combined_small \
  --name comparison_baseline \
  --pattern L1_L2_L3 \
  --display_id -1 \
  --batch_size 8 \
  --num_threads 4 \
  --load_size 320 \
  --crop_size 256 \
  --preprocess crop \
  --n_epochs 25 \
  --n_epochs_decay 25 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 20


python train.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI/combined_small \
  --name comparison_dwt \
  --pattern L1_L2_L3_dwt \
  --display_id -1 \
  --batch_size 8 \
  --num_threads 4 \
  --load_size 320 \
  --crop_size 256 \
  --preprocess crop \
  --n_epochs 25 \
  --n_epochs_decay 25 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 20 \
  --weight_dwt_ll 25 \
  --weight_dwt_detail 25



# Test baseline
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI/combined_small \
  --name comparison_baseline \
  --run_name run_20260107_033500_BCI \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval

# Test DWT
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI/combined_small \
  --name comparison_dwt \
  --run_name run_20260107_033920_BCI
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval


# Evaluate results
python evaluate.py \
  --result_path ./results/comparison_baseline/run_20260107_033500_BCI

python evaluate.py \
  --result_path ./results/comparison_dwt/run_20260107_033920_BCI