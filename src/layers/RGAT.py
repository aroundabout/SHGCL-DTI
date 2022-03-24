import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from dgl.nn.pytorch.conv.gatconv import GATConv
from dgl.nn.pytorch.hetero import HeteroGraphConv


class RGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, args):
        super().__init__()
        # self.conv1 = HeteroGraphConv({
        #     rel: GATConv(in_feats, hid_feats, args.multi_head, feat_drop=args.feat_drop, attn_drop=args.attn_drop,
        #                        residual=args.residual)
        #     for rel in rel_names}, aggregate='mean')
        # self.conv2 = HeteroGraphConv({
        #     rel: GATConv(hid_feats, out_feats, args.multi_head, feat_drop=args.feat_drop,
        #                        attn_drop=args.attn_drop, residual=args.residual)
        #     for rel in rel_names}, aggregate='mean')
        self.conv1 = HeteroGraphConv({
            rel: GATConv(in_feats, hid_feats, args.multi_head)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GATConv(hid_feats, hid_feats, args.multi_head)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = HeteroGraphConv({
            rel: GATConv(hid_feats, out_feats, args.multi_head)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        h = {k: torch.mean(v, (1, 2, 3)) for k, v in h.items()}
        return h
