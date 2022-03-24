import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from dgl.nn.pytorch.hetero import HeteroGraphConv
from dgl.nn.pytorch.conv.graphconv import GraphConv


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, args):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
        self.conv3 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv3(graph, h)
        return h
