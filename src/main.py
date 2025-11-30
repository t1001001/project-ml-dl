import os
import utils.config as conf
from data_prep.get_dataset import get_dataset
from data_prep.split_dataset import split_dataset

def main():
    print("Running the project!")
    get_dataset()
    split_dataset()
    print("Finished running the project!")

if __name__ == "__main__":
    main()