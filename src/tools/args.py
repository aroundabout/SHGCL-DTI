import numpy as np
import torch as th
import torch.nn as nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CoDTI')

    # model
    parser.add_argument("--model", type=str, default='RGCN',
                        help="model before MLP")


    # 公用
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--rounds", type=int, default=1,
                        help="number of training rounds")
    parser.add_argument("--out_dim", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--l2',type=float,default=0.00,
                        help="l2 parameter")
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience.')


    parser.add_argument("--random_walk", type=bool, default=False,
                        help="weather add the random walk edges")
    parser.add_argument("--walk_length", type=int, default=40,
                        help='random walk length')
    parser.add_argument("--sim_matrix", type=bool, default=True,
                        help="weather use the sim martrix of dr and pr")


    # if use rwr
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Restart Probability")


    # attention
    parser.add_argument("--feat_drop", type=float, default=0.0,
                        help="feat dropout")
    parser.add_argument("--attn_drop", type=float, default=0.0,
                        help="attn drop")
    parser.add_argument("--residual", type=bool, default=False,
                        help="residual for attention")
    parser.add_argument("--multi_head", type=int, default=3,
                        help='multi_head')
    # han
    parser.add_argument("--dropout", type=int, default=0.6,
                        help='multi_head')


    return parser.parse_args()
