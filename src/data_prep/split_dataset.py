import shutil
import random
from pathlib import Path
import utils.config as conf

RAW_DIR = Path(conf.DATASET_PATH) / "alessiocorrado99/animals10/versions/2/raw-img"
OUTPUT_DIR = Path(conf.DATASET_PATH) 

translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

def split_dataset():
    print("Splitting the dataset...")

    categories = [f.name for f in RAW_DIR.iterdir() if f.is_dir()]
    print(f"Detected categories: {categories}")

    for category in categories:
        input_folder = RAW_DIR / category
        english_name = translate.get(category, category)

        train_folder = OUTPUT_DIR / "train" / english_name
        test_folder = OUTPUT_DIR / "test" / english_name
        train_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)

        image_files = list(input_folder.glob("*"))
        print(f"{category}: found {len(image_files)} images")
        random.shuffle(image_files)

        split_idx = int(len(image_files) * conf.TRAIN_RATIO)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]

        for img_path in train_files:
            shutil.move(str(img_path), train_folder / img_path.name)
        for img_path in test_files:
            shutil.move(str(img_path), test_folder / img_path.name)

    print("Completed dataset split!")

if __name__ == "__main__":
    split_dataset()