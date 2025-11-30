import shutil
from pathlib import Path
import utils.config as conf

DATASET_PATH = conf.DATASET_PATH

def cleanup():
    folder = Path(DATASET_PATH) / "alessiocorrado99"
    if folder.exists():
        shutil.rmtree(folder)
        print(f"Deleted folder: {folder}")
    else:
        print(f"Folder does not exist: {folder}")

if __name__ == "__main__":
    cleanup()