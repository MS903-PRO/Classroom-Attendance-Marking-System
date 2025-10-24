import os
import cv2
import shutil
import numpy as np
import random

# Define source and output directories
source_dir = 'SWE_DATA'
output_dir = 'SWE_DATA_augmented'
os.makedirs(output_dir, exist_ok=True)

# Desired number of images per student
TARGET_IMAGES_PER_STUDENT = 35  

# Function to apply slight augmentation
def slight_augmentation(image):
    """Applies minor transformations to avoid excessive distortion."""
    # Random brightness adjustment (increase or decrease)
    brightness_factor = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # Small rotation (-5 to 5 degrees)
    h, w = image.shape[:2]
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Small shifts (up to 5% of image size)
    max_shift_x = int(w * 0.05)
    max_shift_y = int(h * 0.05)
    tx = random.randint(-max_shift_x, max_shift_x)
    ty = random.randint(-max_shift_y, max_shift_y)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT)

    return image

# Function to enhance facial features
def enhance_image(image_path, output_path, apply_augmentation=False):
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # Convert back to BGR
    enhanced_img = cv2.merge([enhanced_gray] * 3)

    # Sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)

    # Apply slight augmentation if requested
    if apply_augmentation:
        sharpened = slight_augmentation(sharpened)

    # Save the enhanced image with high quality
    cv2.imwrite(output_path, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])

# Loop through each student folder
for roll in os.listdir(source_dir):
    roll_path = os.path.join(source_dir, roll)
    if os.path.isdir(roll_path):
        # Create corresponding output folder
        out_roll_path = os.path.join(output_dir, roll)
        os.makedirs(out_roll_path, exist_ok=True)

        # Copy and enhance original images
        original_images = [
            file for file in os.listdir(roll_path) 
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]

        for file in original_images:
            src_path = os.path.join(roll_path, file)
            dest_path = os.path.join(out_roll_path, file)

            # Enhance and save the image
            enhance_image(src_path, dest_path, apply_augmentation=False)  # No augmentation for originals

        # Check number of existing images
        existing_images = len([
            file for file in os.listdir(out_roll_path) 
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ])

        # Determine how many new images are needed
        images_needed = max(0, TARGET_IMAGES_PER_STUDENT - existing_images)
        if images_needed == 0:
            continue  

        # Duplicate and enhance more images if needed
        original_images.sort()
        img_index = 0
        while images_needed > 0:
            img_file = original_images[img_index % len(original_images)]
            img_path = os.path.join(roll_path, img_file)

            # Generate new file name
            new_filename = f"enh_{img_index+1}.jpg"
            new_img_path = os.path.join(out_roll_path, new_filename)

            # Enhance and apply slight augmentation
            enhance_image(img_path, new_img_path, apply_augmentation=True)

            images_needed -= 1
            img_index += 1

# Set the root folder path where your subfolders and images are located
root_folder = output_dir

# Function to rename files in each folder
def rename_files_in_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    
    try:
        files = os.listdir(folder_path)
    except PermissionError:
        print(f"Permission denied to access folder: {folder_path}")
        return

    files.sort()

    for i, file in enumerate(files):
        if os.path.isfile(os.path.join(folder_path, file)):
            new_name = f"{folder_name}_{i+1:02d}{os.path.splitext(file)[-1]}"
            old_file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(folder_path, new_name)
            
            if old_file_path != new_file_path:
                try:
                    os.rename(old_file_path, new_file_path)
                except Exception as e:
                    print(f"Error renaming {file}: {e}")

# Walk through all folders (including subfolders) in the root folder
for dirpath, dirnames, filenames in os.walk(root_folder):
    rename_files_in_folder(dirpath)

print("Facial feature enhancement with slight augmentation complete.")

