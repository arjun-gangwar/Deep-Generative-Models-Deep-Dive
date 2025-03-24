import os
import sys
import argparse
import yaml
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

from generative_models.dataset import MnistDataset

def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def main(config):
    # train_data = MnistDataset(args.train_data) 
    # test_data = MnistDataset(args.test_data)
    print(config)
    label_image = pd.read_csv(config["data"]["train_path"])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training script for variational autoencoder")
    parser.add_argument("--config_path", required=True, help="Path to config files.")
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)