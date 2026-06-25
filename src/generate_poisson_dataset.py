import cv2
import numpy as np
import os
import glob
import random
from tqdm import tqdm

def add_poisson_noise(image, photon_count=500):
    """
    Simulates quantum mottle (noise) caused by a low number of X-ray photons.
    Follows a Poisson distribution.
    """
    img_float = image.astype(np.float32) / 255.0
    img_float = np.maximum(img_float, 0.0001) # Avoid zeros for Poisson calculations
    noisy = np.random.poisson(img_float * photon_count) / float(photon_count)
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)

def process_images(train_dir, output_dir, samples_per_class=300, target_size=(256, 256)):
    clean_dir = os.path.join(output_dir, "clean")
    noisy_dir = os.path.join(output_dir, "noisy")
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    
    classes = ["NORMAL", "PNEUMONIA"]
    selected_images = []
    
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        image_paths = glob.glob(os.path.join(cls_dir, "*.*"))
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Randomly sample images
        sampled_paths = random.sample(image_paths, min(samples_per_class, len(image_paths)))
        selected_images.extend([(cls, p) for p in sampled_paths])
        
    print(f"Selected {len(selected_images)} images in total.")
    
    # 10 different noise levels (photon counts). Lower is more noise.
    noise_levels = [10, 20, 50, 100, 150, 200, 300, 500, 750, 1000]
    
    for idx, (cls, path) in enumerate(tqdm(selected_images, desc="Processing images")):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Resize image so all images have the same dimension for the neural network
        img_resized = cv2.resize(img, target_size)
        
        base_name = f"{cls}_img_{idx}"
        
        # Save clean image
        clean_path = os.path.join(clean_dir, f"{base_name}.png")
        cv2.imwrite(clean_path, img_resized)
        
        # Generate and save 1 noisy variant with a randomly selected noise level
        n_level = random.choice(noise_levels)
        noisy_img = add_poisson_noise(img_resized, photon_count=n_level)
        noisy_name = f"{base_name}_noise_{n_level}.png"
        noisy_path = os.path.join(noisy_dir, noisy_name)
        cv2.imwrite(noisy_path, noisy_img)

if __name__ == "__main__":
    TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "poisson_dataset_2")
    
    random.seed(42) # For reproducibility
    process_images(TRAIN_DIR, OUTPUT_DIR, samples_per_class=250)
    print("Dataset generation complete!")
