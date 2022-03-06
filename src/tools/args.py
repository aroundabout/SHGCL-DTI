import numpy as np
import torch as th
import torch.nn as nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='GRDTI')

    parser.add_argument("--epochs", type=int, default=3000,
                        help="number of training epochs")
    parser.add_argument("--rounds", type=int, default=3,
                        help="number of training rounds")
    parser.add_argument("--device", default='cuda',
                        help="cuda or cpu")
    parser.add_argument("--dim-embedding", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of iterations in propagation")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--reg_lambda', type=float, default=1,
                        help="reg_lambda")
    parser.add_argument('--patience', type=int, default=6,
                        help='Early stopping patience.')
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="Restart Probability")
    parser.add_argument("--edge-drop", type=float, default=0.5,
                        help="edge dropout in propagation")

    return parser.parse_args()
