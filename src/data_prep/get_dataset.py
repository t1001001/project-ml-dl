import os
import kagglehub
from utils.config import ROOT

os.environ["KAGGLEHUB_CACHE"] = str(ROOT)

def get_dataset():
    return kagglehub.dataset_download("alessiocorrado99/animals10")

if __name__ == "__main__":
    get_dataset()