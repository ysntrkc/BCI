# Create subset directories
mkdir -p /mnt/c/users/gaming/datasets/BCI/combined_small/train
mkdir -p /mnt/c/users/gaming/datasets/BCI/combined_small/test

# Copy first 100 training images
ls /mnt/c/users/gaming/datasets/BCI/combined/train | head -100 | xargs -I {} cp /mnt/c/users/gaming/datasets/BCI/combined/train/{} /mnt/c/users/gaming/datasets/BCI/combined_small/train/

# Copy all test images (keep test set same for fair comparison)
cp /mnt/c/users/gaming/datasets/BCI/combined/test/* /mnt/c/users/gaming/datasets/BCI/combined_small/test/