import os
import glob
import random
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from train_denoiser import UNet
from generate_poisson_dataset import add_poisson_noise

def test_model(num_images, model_path, train_dir):
    device = torch.device("cpu")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get all image paths
    image_paths = []
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = os.path.join(train_dir, cls)
        paths = glob.glob(os.path.join(cls_dir, "*.*"))
        image_paths.extend([p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
    if len(image_paths) == 0:
        print("No images found in the specified directory.")
        return
        
    # Select random images
    selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    for idx, path in enumerate(selected_paths):
        # Read and resize image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        clean_img_np = cv2.resize(img, (128, 128))
        
        # Add random Poisson noise
        photon_count = random.randint(10, 1000)
        noisy_img_np = add_poisson_noise(clean_img_np, photon_count=photon_count)
        
        # Prepare for model
        noisy_tensor = transform(noisy_img_np).unsqueeze(0).to(device)
        
        # Denoise
        with torch.no_grad():
            output_tensor = model(noisy_tensor)
            
        denoised_img_np = (output_tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        # Calculate Scores
        # PSNR & SSIM
        score_noisy_psnr = psnr(clean_img_np, noisy_img_np)
        score_denoised_psnr = psnr(clean_img_np, denoised_img_np)
        
        score_noisy_ssim = ssim(clean_img_np, noisy_img_np)
        score_denoised_ssim = ssim(clean_img_np, denoised_img_np)
        
        print(f"\nImage {idx+1}/{num_images} (Photon Count: {photon_count}):")
        print(f"  Noisy PSNR: {score_noisy_psnr:.2f} dB, SSIM: {score_noisy_ssim:.4f}")
        print(f"  Denoised PSNR: {score_denoised_psnr:.2f} dB, SSIM: {score_denoised_ssim:.4f}")
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(noisy_img_np, cmap='gray')
        axes[0].set_title(f"Noisy Input (Count: {photon_count})\nPSNR: {score_noisy_psnr:.2f}, SSIM: {score_noisy_ssim:.2f}")
        axes[0].axis('off')
        
        axes[1].imshow(denoised_img_np, cmap='gray')
        axes[1].set_title(f"Denoised Output\nPSNR: {score_denoised_psnr:.2f}, SSIM: {score_denoised_ssim:.2f}")
        axes[1].axis('off')
        
        axes[2].imshow(clean_img_np, cmap='gray')
        axes[2].set_title("Original Image")
        axes[2].axis('off')
        
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results_2", f"test_result_{idx+1}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Denoising Autoencoder")
    parser.add_argument("--num_images", type=int, default=5, help="Number of random images to test")
    
    default_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "unet_denoiser_model.pth")
    parser.add_argument("--model", type=str, default=default_model, help="Path to trained model weights")
    
    default_train = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train")
    parser.add_argument("--train_dir", type=str, default=default_train, help="Path to original images")
    
    args = parser.parse_args()
    
    test_model(args.num_images, args.model, args.train_dir)
