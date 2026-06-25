import cv2
import numpy as np
import os
import glob

def add_gaussian_noise(image, mean=0, std=25):
    """
    Simulates electronic sensor noise using a Gaussian distribution.
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_poisson_noise(image, photon_count=500):
    """
    Simulates quantum mottle (noise) caused by a low number of X-ray photons.
    Follows a Poisson distribution.
    """
    img_float = image.astype(np.float32) / 255.0
    img_float = np.maximum(img_float, 0.0001) # Avoid zeros for Poisson calculations
    noisy = np.random.poisson(img_float * photon_count) / float(photon_count)
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, prob=0.02):
    """
    Simulates dead/stuck detector pixels (impulse noise).
    'prob' is the probability of a pixel being affected.
    """
    noisy_image = np.copy(image)
    
    # Calculate amount of salt and pepper
    num_salt = np.ceil(prob * image.size * 0.5)
    num_pepper = np.ceil(prob * image.size * 0.5)
    
    # Add Salt (White pixels)
    coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords_salt)] = 255

    # Add Pepper (Black pixels)
    coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords_pepper)] = 0
    
    return noisy_image

def add_speckle_noise(image, std=0.2):
    """
    Adds multiplicative speckle noise to the image.
    """
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image.astype(np.float32) + (image.astype(np.float32) * noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def process_images(input_dir, output_dir, num_images=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get all image files
    image_paths = glob.glob(os.path.join(input_dir, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    # Limit the number of images if specified
    if num_images is not None:
        image_paths = image_paths[:num_images]
        print(f"Processing {len(image_paths)} images...")
    else:
        print(f"Processing all {len(image_paths)} images...")
        
    for i, path in enumerate(image_paths):
        # Read as grayscale since X-rays are grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {path}, skipping.")
            continue
            
        base_name = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1]
        
        # 1. Gaussian Noise
        gaussian_img = add_gaussian_noise(img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_gaussian{ext}"), gaussian_img)
        
        # 2. Poisson Noise
        poisson_img = add_poisson_noise(img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_poisson{ext}"), poisson_img)
        
        # 3. Salt and Pepper Noise
        sp_img = add_salt_and_pepper_noise(img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_saltpepper{ext}"), sp_img)
        
        # 4. Speckle Noise
        speckle_img = add_speckle_noise(img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_speckle{ext}"), speckle_img)
        
        print(f"Processed {i+1}/{len(image_paths)}: {base_name}")
    
    print("Done! Noise-augmented images have been saved.")

if __name__ == "__main__":
    # ==========================================
    #             CONFIGURATION
    # ==========================================
    
    # Put the path to your folder containing the original X-Rays here
    INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train", "NORMAL")
    
    # Put the path where you want the noisy images to be saved
    OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "xrays_with_noise")
    
    # DECIDE HOW MANY IMAGES TO PROCESS:
    # Set to an integer (e.g., 5) to only process 5 images.
    # Set to None to process EVERY image in the folder.
    NUMBER_OF_IMAGES = 5
    
    # ==========================================
    
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, NUMBER_OF_IMAGES)