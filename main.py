import torch
from torch.utils.data import DataLoader

import os, subprocess

from config import *
from dataset.hme_dataset import HMEDataset



def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # create model
    
    root_dir = download_data()
    print(root_dir)
    # init dataset

    train_dataset   = HMEDataset(root_dir, "train")
    valid_dataset   = HMEDataset(root_dir, "valid")
    test_dataset    = HMEDataset(root_dir, "test")
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)



def download_data(url="https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"):
    filename = url.split("/")[-1]
    dirname = filename.split('.')[0]

    # download the file if it doesn't already exist
    if not os.path.exists(dirname):
        subprocess.run(["wget", "-nc", url], check=True)
        subprocess.run(["tar", "zxf", filename], check=True)
    else: print(f"{filename} already exists. Skipping download.")

    # extract the archive
    
    
    if os.path.exists(filename):os.remove(filename)
    else:print(f"Tar file {filename} not found for deletion.")


    return dirname


if __name__ == "__main__":
    main()