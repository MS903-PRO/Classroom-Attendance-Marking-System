import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import sys

# Ensure a roll number is provided
if len(sys.argv) < 2:
    print("No roll number provided.")
    sys.exit(1)

roll = sys.argv[1]  # Get the roll number from the command line

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define source and destination directories
source_dir = 'SWE_DATA'
output_dir = 'SWE_DATA_augmented'
os.makedirs(output_dir, exist_ok=True)

# Define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Desired number of images per student
TARGET_IMAGES_PER_STUDENT = 35  

# Process only the specified roll number
roll_path = os.path.join(source_dir, roll)

if os.path.isdir(roll_path):
    # Create corresponding output folder
    out_roll_path = os.path.join(output_dir, roll)
    os.makedirs(out_roll_path, exist_ok=True)

    # Copy original images to augmented folder
    original_images = [
        file for file in os.listdir(roll_path) 
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ]

    for file in original_images:
        src_path = os.path.join(roll_path, file)
        dest_path = os.path.join(out_roll_path, file)

        if not os.path.exists(dest_path):  # Avoid duplicate copies
            shutil.copy2(src_path, dest_path)

    # Check number of existing images
    existing_images = len([
        file for file in os.listdir(out_roll_path) 
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])

    # Determine how many new images are needed
    images_needed = max(0, TARGET_IMAGES_PER_STUDENT - existing_images)
    if images_needed == 0:
        print(f"Skipping '{roll}', already has {TARGET_IMAGES_PER_STUDENT} images.")
    else:
        # Shuffle original images for variety in augmentation
        random.shuffle(original_images)

        # Process images for augmentation
        img_index = 0
        while images_needed > 0:
            img_file = original_images[img_index % len(original_images)]  # Cycle through images
            img_path = os.path.join(roll_path, img_file)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Create an iterator for augmentation
            aug_iter = datagen.flow(
                x,
                batch_size=1,
                save_to_dir=out_roll_path,
                save_prefix='aug',
                save_format='jpg'
            )

            # Generate one image per loop iteration
            next(aug_iter)
            images_needed -= 1
            img_index += 1  # Move to the next image

        print(f"'{roll}' now has {TARGET_IMAGES_PER_STUDENT} images (original + augmented).")

else:
    print(f"Roll number '{roll}' not found in {source_dir}. No augmentation performed.")

# Set the root folder path where your subfolders and images are located
root_folder = os.path.join(output_dir, roll)  # Process only this roll's folder

# Function to rename files in the specified folder
def rename_files_in_folder(folder_path):
    """Renames all files in the given folder with the format roll_number_01.jpg."""
    folder_name = os.path.basename(folder_path)
    
    try:
        files = sorted(os.listdir(folder_path))  # Get files and sort them
    except PermissionError:
        print(f"Permission denied to access folder: {folder_path}")
        return

    # Rename each file
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

# Rename files only in the specified roll's folder
rename_files_in_folder(root_folder)

print(f"Data augmentation complete for roll number: {roll}")

