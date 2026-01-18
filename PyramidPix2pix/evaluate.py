import os
import cv2 as cv
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
import argparse
import torch
from torchvision import models, transforms
from scipy import linalg
from scipy.spatial.distance import jensenshannon


def parse_opt():
    # Set train options
    parser = argparse.ArgumentParser(description="Evaluate options")
    parser.add_argument(
        "--result_path",
        type=str,
        default="./results/pyramidpix2pix",
        help="results saved path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="device to use for feature extraction",
    )
    parser.add_argument(
        "--kid_batch_size",
        type=int,
        default=100,
        help="batch size for KID kernel computation (lower = less memory)",
    )
    opt = parser.parse_args()
    return opt


class InceptionFeatureExtractor:
    """Extract features from InceptionV3 for FID and KID calculation"""

    def __init__(self, device="cuda"):
        self.device = device
        # Load InceptionV3 pre-trained model
        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
        )
        self.model.fc = torch.nn.Identity()  # Remove final classification layer
        self.model.eval()
        self.model.to(device)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def extract_features(self, images):
        """Extract features from a list of image paths"""
        features = []
        for img_path in tqdm(images, desc="Extracting features"):
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            feat = self.model(img_tensor)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)


def calculate_fid(real_features, fake_features, eps=1e-6):
    """
    Calculate Fréchet Inception Distance (FID)
    Lower is better
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # Calculate squared difference of means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Add epsilon to diagonal for numerical stability
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    # Calculate sqrt of product between covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # Check for imaginary numbers (numerical errors)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} too large in covmean")
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    """Polynomial kernel for KID calculation"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (gamma * np.dot(X, Y.T) + coef0) ** degree
    return K


def calculate_kid_batched(real_features, fake_features, batch_size=100):
    """
    Calculate Kernel Inception Distance (KID) with batched computation
    Lower is better
    Uses unbiased estimator by excluding diagonal elements
    Uses all available data for accurate evaluation
    
    Batching is done for memory efficiency but computes the exact same result
    as non-batched version by accumulating kernel sums across batches.
    
    Args:
        real_features: Features from real images (n_real x feature_dim)
        fake_features: Features from fake images (n_fake x feature_dim)
        batch_size: Batch size for kernel computation (default: 100)
    
    Returns:
        KID score (float)
    """
    n_real = len(real_features)
    n_fake = len(fake_features)
    
    print(f"Computing KID with batched kernels (batch_size={batch_size})...")
    print(f"Real features: {n_real}, Fake features: {n_fake}")

    # Accumulate kernel sums
    sum_k_rr = 0.0
    sum_k_ff = 0.0
    sum_k_rf = 0.0
    
    # Compute k_rr (real vs real) in batches
    print("Computing k_rr (real vs real)...")
    for i in tqdm(range(0, n_real, batch_size), desc="k_rr batches"):
        end_i = min(i + batch_size, n_real)
        batch_real_i = real_features[i:end_i]
        
        for j in range(0, n_real, batch_size):
            end_j = min(j + batch_size, n_real)
            batch_real_j = real_features[j:end_j]
            
            k_batch = polynomial_kernel(batch_real_i, batch_real_j)
            
            # Exclude diagonal elements (only when i == j)
            if i == j:
                # Zero out diagonal elements within this batch
                for k in range(len(k_batch)):
                    k_batch[k, k] = 0
            
            sum_k_rr += np.sum(k_batch)
    
    # Compute k_ff (fake vs fake) in batches
    print("Computing k_ff (fake vs fake)...")
    for i in tqdm(range(0, n_fake, batch_size), desc="k_ff batches"):
        end_i = min(i + batch_size, n_fake)
        batch_fake_i = fake_features[i:end_i]
        
        for j in range(0, n_fake, batch_size):
            end_j = min(j + batch_size, n_fake)
            batch_fake_j = fake_features[j:end_j]
            
            k_batch = polynomial_kernel(batch_fake_i, batch_fake_j)
            
            # Exclude diagonal elements (only when i == j)
            if i == j:
                # Zero out diagonal elements within this batch
                for k in range(len(k_batch)):
                    k_batch[k, k] = 0
            
            sum_k_ff += np.sum(k_batch)
    
    # Compute k_rf (real vs fake) in batches
    print("Computing k_rf (real vs fake)...")
    for i in tqdm(range(0, n_real, batch_size), desc="k_rf batches"):
        end_i = min(i + batch_size, n_real)
        batch_real = real_features[i:end_i]
        
        for j in range(0, n_fake, batch_size):
            end_j = min(j + batch_size, n_fake)
            batch_fake = fake_features[j:end_j]
            
            k_batch = polynomial_kernel(batch_real, batch_fake)
            sum_k_rf += np.sum(k_batch)
    
    # Calculate KID with unbiased estimator
    # For k_rr and k_ff, we excluded diagonal so divide by n*(n-1)
    # For k_rf, all elements are used so divide by n_real*n_fake
    kid = (sum_k_rr / (n_real * (n_real - 1)) + 
           sum_k_ff / (n_fake * (n_fake - 1)) - 
           2 * sum_k_rf / (n_real * n_fake))

    return kid


def calculate_jsd(real_images, fake_images):
    """
    Calculate Jensen-Shannon Divergence (JSD) on pixel histograms
    Lower is better (0 = identical distributions, 1 = completely different)
    """
    jsd_values = []

    for real_path, fake_path in tqdm(
        zip(real_images, fake_images), total=len(real_images), desc="Calculating JSD"
    ):
        real = cv.imread(real_path)
        fake = cv.imread(fake_path)

        if real is None or fake is None:
            continue

        # Calculate normalized histograms for each channel
        for channel in range(3):  # BGR channels
            real_hist, _ = np.histogram(
                real[:, :, channel].flatten(), bins=256, range=(0, 256), density=True
            )
            fake_hist, _ = np.histogram(
                fake[:, :, channel].flatten(), bins=256, range=(0, 256), density=True
            )

            # Add small epsilon to avoid log(0)
            real_hist = real_hist + 1e-10
            fake_hist = fake_hist + 1e-10

            # Normalize to ensure they sum to 1
            real_hist = real_hist / real_hist.sum()
            fake_hist = fake_hist / fake_hist.sum()

            # Calculate JSD
            jsd = jensenshannon(real_hist, fake_hist) ** 2
            jsd_values.append(jsd)

    return np.mean(jsd_values) if jsd_values else 0.0


def evaluate_metrics(result_path, device="cuda", kid_batch_size=100):
    """
    Main evaluation function to calculate all metrics
    
    Args:
        result_path: Path to results directory
        device: Device to use for feature extraction ('cuda' or 'cpu')
        kid_batch_size: Batch size for KID kernel computation
    
    Returns:
        Dictionary containing all computed metrics
    """
    psnr = []
    ssim = []
    fake_images = []
    real_images = []

    # Collect image paths and calculate PSNR/SSIM
    print("Calculating PSNR and SSIM...")
    image_dir = os.path.join(result_path, "test_latest/images")
    
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    for i in tqdm(os.listdir(image_dir), desc="Processing images"):
        if "fake_B" in i:
            try:
                fake_path = os.path.join(image_dir, i)
                real_path = os.path.join(image_dir, i.replace("fake_B", "real_B"))

                if not os.path.exists(real_path):
                    print(f"Warning: Real image not found for {i}")
                    continue

                fake = cv.imread(fake_path)
                real = cv.imread(real_path)

                if fake is None or real is None:
                    print(f"Warning: Could not read images for {i}")
                    continue

                PSNR = peak_signal_noise_ratio(fake, real)
                psnr.append(PSNR)

                SSIM = structural_similarity(
                    fake, real, multichannel=True, channel_axis=2
                )
                ssim.append(SSIM)

                # Store paths for FID, KID, and JSD calculation
                fake_images.append(fake_path)
                real_images.append(real_path)
            except Exception as e:
                print(f"Error processing {i}: {e}")
                continue

    if len(psnr) == 0 or len(ssim) == 0:
        raise ValueError("No valid images were evaluated!")

    average_psnr = sum(psnr) / len(psnr)
    average_ssim = sum(ssim) / len(ssim)
    
    print(f"\nProcessed {len(fake_images)} image pairs")
    print(f"Average PSNR: {average_psnr:.6f}")
    print(f"Average SSIM: {average_ssim:.6f}")

    # Initialize results dictionary
    results = {
        "psnr": average_psnr,
        "ssim": average_ssim,
        "num_images": len(fake_images),
    }

    # Calculate FID, KID, and JSD
    print("\nCalculating FID, KID, and JSD metrics...")
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        device = "cpu"
    
    print(f"Using device: {device}")

    try:
        # Extract features for FID and KID
        feature_extractor = InceptionFeatureExtractor(device=device)
        print("Extracting features from real images...")
        real_features = feature_extractor.extract_features(real_images)
        print("Extracting features from fake images...")
        fake_features = feature_extractor.extract_features(fake_images)

        print(f"Feature shape: {real_features.shape}")

        # Calculate FID
        print("\nCalculating FID...")
        fid_score = calculate_fid(real_features, fake_features)
        print(f"FID: {fid_score:.4f}")
        results["fid"] = fid_score

        # Calculate KID
        print("Calculating KID...")
        kid_score = calculate_kid_batched(real_features, fake_features, batch_size=kid_batch_size)
        print(f"KID: {kid_score:.6f}")
        results["kid"] = kid_score

        # Calculate JSD
        print("Calculating JSD...")
        jsd_score = calculate_jsd(real_images, fake_images)
        print(f"JSD: {jsd_score:.6f}")
        results["jsd"] = jsd_score

    except Exception as e:
        print(f"Error calculating FID/KID/JSD: {e}")
        import traceback
        traceback.print_exc()
        print("\nContinuing with PSNR and SSIM only...")

    return results


def print_summary(results):
    """Print evaluation summary in a nice format"""
    print("\n" + "=" * 60)
    print(" " * 20 + "EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Number of images:  {results['num_images']}")
    print("-" * 60)
    print(f"Average PSNR:      {results['psnr']:.6f} dB")
    print(f"Average SSIM:      {results['ssim']:.6f}")
    if "fid" in results:
        print(f"FID Score:         {results['fid']:.6f}")
    if "kid" in results:
        print(f"KID Score:         {results['kid']:.6f}")
    if "jsd" in results:
        print(f"JSD Score:         {results['jsd']:.6f}")
    print("=" * 60)
    print("\nNote: Lower values are better for FID, KID, and JSD")
    print("      Higher values are better for PSNR and SSIM")
    print("=" * 60)


def main():
    """Main function to run evaluation"""
    opt = parse_opt()
    
    print("=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    print(f"Result path: {opt.result_path}")
    print(f"Device: {opt.device}")
    print(f"KID batch size: {opt.kid_batch_size}")
    print("=" * 60 + "\n")
    
    try:
        results = evaluate_metrics(
            result_path=opt.result_path,
            device=opt.device,
            kid_batch_size=opt.kid_batch_size
        )
        print_summary(results)
        
        # Save results to file
        results_file = os.path.join(opt.result_path, "evaluation_results.txt")
        with open(results_file, "w") as f:
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
