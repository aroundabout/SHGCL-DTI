import numpy as np
import torch as th
import torch.nn as nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CoDTI')

    # model
    parser.add_argument("--model", type=str, default='RGAT',
                        help="model before MLP")
    parser.add_argument("--device", type=str, default='cuda:3')
    # 公用
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--rounds", type=int, default=1,
                        help="number of training rounds")
    parser.add_argument("--out_dim", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.0008,
                        help="learning rate")
    parser.add_argument('--l2', type=float, default=0.00,
                        help="l2 parameter")
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience.')

    parser.add_argument("--random_walk", type=bool, default=False,
                        help="weather add the random walk edges")
    parser.add_argument("--walk_length", type=int, default=40,
                        help='random walk length')
    parser.add_argument("--sim_matrix", type=bool, default=True,
                        help="weather use the sim martrix of dr and pr")

    # attention
    parser.add_argument("--feat_drop", type=float, default=0.3,
                        help="feat dropout")
    parser.add_argument("--attn_drop", type=float, default=0.3,
                        help="attn drop")
    parser.add_argument("--residual", type=bool, default=True,
                        help="residual for attention")
    parser.add_argument("--multi_head", type=int, default=3,
                        help='multi_head')
    # HECO
    parser.add_argument("--tau", type=float, default=0.5,
                        help='multi_head')
    parser.add_argument("--lam", type=int, default=0.5,
                        help='multi_head')
    parser.add_argument("--epoch_mlp", type=int, default=500,
                        help="number of training epochs")
    return parser.parse_args()


def parse_argsCO():
    parser = argparse.ArgumentParser(description='CoDTI')
    parser.add_argument("--device", type=str, default='cuda:3')

    # 公用
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--rounds", type=int, default=1,
                        help="number of training rounds")
    parser.add_argument("--out_dim", type=int, default=512,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--l2', type=float, default=0.0,
                        help="l2 parameter")
    parser.add_argument('--patience', type=int, default=40,
                        help='Early stopping patience.')

    parser.add_argument("--feat_drop", type=float, default=0.3,
                        help="feat dropout")
    parser.add_argument("--attn_drop", type=float, default=0.3,
                        help="attn drop")
    parser.add_argument("--residual", type=bool, default=True,
                        help="residual for attention")
    parser.add_argument("--multi_head", type=int, default=3,
                        help='multi_head')
    # HECO
    parser.add_argument("--tau", type=float, default=0.5,
                        help='')
    parser.add_argument("--lam", type=int, default=0.5,
                        help='')
    parser.add_argument("--epoch_mlp", type=int, default=1000,
                        help="number of MLP training epochs")
    return parser.parse_args()
