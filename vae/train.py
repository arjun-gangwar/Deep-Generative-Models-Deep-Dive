import os
import sys
import argparse

import torch
from torch import nn
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

from generative_models.dataset import MnistDataset

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self):
        pass

def main(args):
    train_data = MnistDataset(args.train_data) 
    test_data = MnistDataset(args.test_data)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training script for variational autoencoder")
    parser.add_argument("--train_data", default="/speech/arjun/exps/1study/generative_models/data/MNIST_CSV/mnist_train.csv", help="path to train csv")
    parser.add_argument("--test_data", default="/speech/arjun/exps/1study/generative_models/data/MNIST_CSV/mnist_test.csv", help="path to test csv")
    args = parser.parse_args()
    main(args)