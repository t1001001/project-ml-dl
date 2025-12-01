import os
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import utils.config as conf

TRAIN_FOLDERS = [
    conf.DATASET_TRAIN_BUTTERFLY_PATH,
    conf.DATASET_TRAIN_CAT_PATH,
    conf.DATASET_TRAIN_CHICKEN_PATH,
    conf.DATASET_TRAIN_COW_PATH,
    conf.DATASET_TRAIN_DOG_PATH,
    conf.DATASET_TRAIN_ELEPHANT_PATH,
    conf.DATASET_TRAIN_HORSE_PATH,
    conf.DATASET_TRAIN_SHEEP_PATH,
    conf.DATASET_TRAIN_SPIDER_PATH,
    conf.DATASET_TRAIN_SQUIRREL_PATH,
]
TEST_FOLDERS = [
    conf.DATASET_TEST_BUTTERFLY_PATH,
    conf.DATASET_TEST_CAT_PATH,
    conf.DATASET_TEST_CHICKEN_PATH,
    conf.DATASET_TEST_COW_PATH,
    conf.DATASET_TEST_DOG_PATH,
    conf.DATASET_TEST_ELEPHANT_PATH,
    conf.DATASET_TEST_HORSE_PATH,
    conf.DATASET_TEST_SHEEP_PATH,
    conf.DATASET_TEST_SPIDER_PATH,
    conf.DATASET_TEST_SQUIRREL_PATH,
]

BACKUP_FOLDERS = {
    conf.DATASET_TRAIN_BUTTERFLY_PATH: conf.DATASET_BACKUP_BUTTERFLY,
    conf.DATASET_TRAIN_CAT_PATH: conf.DATASET_BACKUP_CAT,
    conf.DATASET_TRAIN_CHICKEN_PATH: conf.DATASET_BACKUP_CHICKEN,
    conf.DATASET_TRAIN_COW_PATH: conf.DATASET_BACKUP_COW,
    conf.DATASET_TRAIN_DOG_PATH: conf.DATASET_BACKUP_DOG,
    conf.DATASET_TRAIN_ELEPHANT_PATH: conf.DATASET_BACKUP_ELEPHANT,
    conf.DATASET_TRAIN_HORSE_PATH: conf.DATASET_BACKUP_HORSE,
    conf.DATASET_TRAIN_SHEEP_PATH: conf.DATASET_BACKUP_SHEEP,
    conf.DATASET_TRAIN_SPIDER_PATH: conf.DATASET_BACKUP_SPIDER,
    conf.DATASET_TRAIN_SQUIRREL_PATH: conf.DATASET_BACKUP_SQUIRREL
}

ROTATIONS_PER_IMAGE = conf.ROTATIONS_PER_IMAGE
ROTATION_ANGLE_RANGE = conf.ROTATION_ANGLE_RANGE  
BRIGHTNESS_ADJUSTMENTS_PER_IMAGE = conf.BRIGHTNESS_ADJUSTMENTS_PER_IMAGE
BRIGHTNESS_RANGE = conf.BRIGHTNESS_RANGE
GAUSSIAN_BLUR_PER_IMAGE = conf.GAUSSIAN_BLUR_PER_IMAGE
GAUSSIAN_RANDOM_BLUR_KERNEL = conf.GAUSSIAN_RANDOM_BLUR_KERNEL  
NOISE_PER_IMAGE = conf.NOISE_PER_IMAGE
NOISE_NOISE_INTENSITY = conf.NOISE_NOISE_INTENSITY  
COLOR_JITTER_VARIANTS = conf.COLOR_JITTER_VARIANTS


def get_file_extension(image_path):
    """Returns the original file extension."""
    return Path(image_path).suffix

def count_images(folder):
    """Counts the number of images in a folder."""
    return len(list(Path(folder).rglob("*"))) + len(list(Path(folder).rglob("*")))

def log_dataset_size(folder, stage):
    """Logs the number of images before and after augmentation."""
    num_images = count_images(folder)
    print(f"[INFO] {stage} - {folder}: {num_images} images")

def convert_to_grayscale(image_path, save_path):
    """Convert an image to grayscale and save it."""
    if os.path.exists(save_path):
        return
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path, grayscale_image)

def rotate_image(image_path, save_dir):
    """Create multiple rotated versions of the image and save them."""
    image = Image.open(image_path).convert("L")
    filename = Path(image_path).stem
    ext = get_file_extension(image_path)
    for i in range(ROTATIONS_PER_IMAGE):
        rotated_path = os.path.join(save_dir, f"{filename}_rotated_{i}{ext}")
        if os.path.exists(rotated_path):
            continue
        angle = random.uniform(*ROTATION_ANGLE_RANGE)
        rotated_image = image.rotate(angle, resample=Image.BICUBIC)
        rotated_image.save(rotated_path)

def adjust_brightness(image_path, save_dir):
    """Create multiple brightness-adjusted versions of the image and save them."""
    image = Image.open(image_path).convert("L")
    filename = Path(image_path).stem
    ext = get_file_extension(image_path)
    enhancer = ImageEnhance.Brightness(image)
    for i in range(BRIGHTNESS_ADJUSTMENTS_PER_IMAGE):
        bright_path = os.path.join(save_dir, f"{filename}_brightness_{i}{ext}")
        if os.path.exists(bright_path):
            continue
        factor = random.uniform(*BRIGHTNESS_RANGE)
        bright_image = enhancer.enhance(factor)
        bright_image.save(bright_path)

def add_gaussian_blur(image_path, save_dir):
    """Apply Gaussian blur multiple times with different kernel sizes."""
    image = cv2.imread(image_path)
    filename = Path(image_path).stem
    ext = get_file_extension(image_path)
    for i in range(GAUSSIAN_BLUR_PER_IMAGE):
        blur_size = random.choice(GAUSSIAN_RANDOM_BLUR_KERNEL)  
        blur_path = os.path.join(save_dir, f"{filename}_blurred_{i}{ext}")
        if not os.path.exists(blur_path):
            blurred_image = cv2.GaussianBlur(image, blur_size, 0)
            cv2.imwrite(blur_path, blurred_image)

def add_noise(image_path, save_dir):
    """Apply noise multiple times with different intensities."""
    image = cv2.imread(image_path)
    filename = Path(image_path).stem
    ext = get_file_extension(image_path)
    for i in range(NOISE_PER_IMAGE):
        noise_path = os.path.join(save_dir, f"{filename}_noisy_{i}{ext}")
        if os.path.exists(noise_path):
            continue
        noise_intensity = random.randint(*NOISE_NOISE_INTENSITY)  
        noise = np.random.normal(0, noise_intensity, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        cv2.imwrite(noise_path, noisy_image)

def flip_image(image_path, save_dir):
    """Flip an image horizontally and vertically and save it."""
    image = cv2.imread(image_path)
    filename = Path(image_path).stem
    ext = get_file_extension(image_path)
    flipped_images = [
        (cv2.flip(image, 1), f"_flipped_h{ext}"),  
        (cv2.flip(image, 0), f"_flipped_v{ext}"),  
    ]
    for img, suffix in flipped_images:
        flip_path = os.path.join(save_dir, f"{filename}{suffix}")
        if os.path.exists(flip_path): 
            continue
        cv2.imwrite(flip_path, img)

def apply_clahe(image_path, save_path):
    """Apply CLAHE and save the image."""
    if os.path.exists(save_path):
        return
    image = cv2.imread(image_path)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced_image = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(save_path, enhanced_image)

def apply_color_jitter(image_path, save_dir):
    """Apply multiple random color jitter variations and save them."""
    image = Image.open(image_path).convert("L")  
    filename = Path(image_path).stem
    ext = get_file_extension(image_path)
    jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1)
    for i in range(COLOR_JITTER_VARIANTS):
        jittered_image = jitter(image)
        jitter_path = os.path.join(save_dir, f"{filename}_colorjitter_{i}{ext}")
        if not os.path.exists(jitter_path):
            jittered_image.save(jitter_path)

def process_images(folder, apply_augmentation=True):
    """Convert images to grayscale and optionally apply augmentation."""
    print("--- Starting data augmentation ---")
    backup_folder = BACKUP_FOLDERS.get(folder)
    if backup_folder is None:
        backup_folder = os.path.join("datasets", "backup", os.path.basename(folder))
    os.makedirs(backup_folder, exist_ok=True)
    log_dataset_size(folder, "Before Augmentation")
    image_files = list(Path(folder).rglob("*")) + list(Path(folder).rglob("*"))
    for image_path in tqdm(image_files, desc=f"Processing {folder}"):
        image_path = str(image_path)
        ext = get_file_extension(image_path)
        grayscale_save_path = image_path.replace(ext, f"_grayscale{ext}")
        convert_to_grayscale(image_path, grayscale_save_path)
        new_backup_path = os.path.join(backup_folder, os.path.basename(image_path))
        if not os.path.exists(new_backup_path):
            os.rename(image_path, new_backup_path)
        if apply_augmentation:
            rotate_image(grayscale_save_path, folder)
            adjust_brightness(grayscale_save_path, folder)
            add_gaussian_blur(grayscale_save_path, folder)
            add_noise(grayscale_save_path, folder)
            flip_image(grayscale_save_path, folder)
            apply_clahe(grayscale_save_path, folder)
            apply_color_jitter(grayscale_save_path, folder)
    log_dataset_size(folder, "After Augmentation")
    print("--- Finished data augmentation ---")

if __name__ == "__main__":
    for folder in TRAIN_FOLDERS:
        process_images(folder, apply_augmentation=True)