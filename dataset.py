import os
import sys
import csv
import argparse
import pandas as pd

import torch
from torch.utils.data import Dataset

class MnistDataset(Dataset):
    def __init__(self, csv_path):
        self.label_image = pd.read_csv(csv_path)
        self.labels = self.label_image.iloc[:, 0]
        self.images = self.label_image.iloc[:, 1:]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = torch.Tensor(self.images.iloc[idx, :]).reshape(28,28)
        label = torch.Tensor(self.labels.iloc[idx])
        return image, label

def main():
    pass

if __name__ == "__main__":
    main()