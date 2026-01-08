# combine two folders HE and IHC into one folder combined
python datasets/combine_A_and_B.py --fold_A /mnt/c/users/gaming/datasets/BCI_dataset/HE --fold_B /mnt/c/users/gaming/datasets/BCI_dataset/IHC --fold_AB /mnt/c/users/gaming/datasets/BCI_dataset/combined

# train the model on combined dataset
python train.py --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined --gpu_ids 0 --pattern L1_L2_L3_L4 --display_id -1 --preprocess crop --crop_size 256

# test the model
python test.py --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined --gpu_ids 0 --crop_size 256

# evaluate the results
python evaluate.py --result_path ./results/pyramidpix2pix

python train.py --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined --gpu_ids 0 --pattern L1_L2 --display_id -1 --preprocess crop --crop_size 512

python train.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined \
  --gpu_ids 0 \
  --pattern L1_L2_L3 \
  --display_id -1 \
  --batch_size 8 \
  --num_threads 8 \
  --load_size 320 \
  --preprocess scale_width \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50

nohup python train.py --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined --gpu_ids 0 --pattern L1_L2_L3 --display_id -1 --batch_size 8 --num_threads 8 --crop_size 512 --preprocess crop --ngf 64 --ndf 64 --n_epochs 50 --n_epochs_decay 50 --lr 0.0002 --save_epoch_freq 10 --print_freq 50 > training_output.txt 2>&1 &


python train.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined \
  --gpu_ids 0 \
  --pattern L1_L2_L3_L4 \
  --display_id -1 \
  --batch_size 8 \
  --num_threads 8 \
  --load_size 320 \
  --preprocess scale_width \
  --ngf 64 \
  --ndf 64 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --lr 0.0002 \
  --save_epoch_freq 10 \
  --print_freq 50 \
  --continue_train \
  --epoch_count 10

  python test.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined \
  --gpu_ids 0 \
  --preprocess scale_width \
  --load_size 320 \
  --eval


nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined \
  --gpu_ids 0 \
  --pattern L1_L2_L3 \
  --display_id -1 \
  --batch_size 8 \
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
  > training_output.txt 2>&1 &

# Run 002 2025-12-28 with BCI dataset L1_L2_L3 pattern
nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined \
  --gpu_ids 0 \
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
  > training_output.txt 2>&1 &
  
python test.py \
  --dataroot /mnt/c/users/gaming/datasets/BCI_dataset/combined \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval
  
The average psnr is 21.973196274676656
The average ssim is 0.5470951924758315


# Run 003 2025-12-30 with MIST HER2 dataset
python datasets/combine_A_and_B.py --fold_A /mnt/c/users/gaming/datasets/MIST/HER2/HE --fold_B /mnt/c/users/gaming/datasets/MIST/HER2/IHC --fold_AB /mnt/c/users/gaming/datasets/MIST/HER2/combined

nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
  --gpu_ids 0 \
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
  > training_output.txt 2>&1 &

python test.py \
  --dataroot /mnt/c/users/gaming/datasets/MIST/HER2/combined \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval

The average psnr is 15.060739531900422
The average ssim is 0.24749067621827142

# Run 004 2025-12-31 with HER2match dataset
python datasets/combine_A_and_B.py --fold_A /mnt/c/users/gaming/datasets/HER2match/HE --fold_B /mnt/c/users/gaming/datasets/HER2match/IHC --fold_AB /mnt/c/users/gaming/datasets/HER2match/combined

nohup python train.py \
  --dataroot /mnt/c/users/gaming/datasets/HER2match/combined \
  --gpu_ids 0 \
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
  > training_output.txt 2>&1 &

python test.py \
  --dataroot /mnt/c/users/gaming/datasets/HER2match/combined \
  --preprocess crop \
  --load_size 320 \
  --crop_size 256 \
  --eval

python evaluate.py --result_path ./results/pyramidpix2pix

The average psnr is 19.32617032458822
The average ssim is 0.4215733808030972