import shutil
from pathlib import Path
import utils.config as conf

def cleanup():
    folder = Path(conf.DATASET_PATH) / "alessiocorrado99"
    if folder.exists():
        shutil.rmtree(folder)
        print(f"Deleted folder: {folder}")
    else:
        print(f"Folder does not exist: {folder}")

if __name__ == "__main__":
    cleanup()