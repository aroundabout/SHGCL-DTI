import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch.conv.gatconv import GATConv
from dgl.nn.pytorch.conv.graphconv import GraphConv


class RelGraphConvLayer(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.2,
                 num_heads=3,
                 con="RGCN"):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.con = con
        if con == "RGAT":
            self.conv = dgl.nn.pytorch.HeteroGraphConv({
                rel: dgl.nn.pytorch.GATConv(in_feat, out_feat, num_heads=num_heads, residual=True, bias=True)
                for rel in rel_names
            })
        else:
            self.conv = dgl.nn.pytorch.HeteroGraphConv({
                rel: dgl.nn.pytorch.GraphConv(in_feat, out_feat, norm='both', weight=True, bias=True)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dgl.nn.pytorch.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        g = g.local_var()
        if self.use_weight and self.conv == "RGCN":
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)
        if self.con == "RGAT":
            hs = {k: th.mean(v, 1) for k, v in hs.items()}

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, args):
        super().__init__()

        self.conv1 = RelGraphConvLayer(in_feats, hid_feats, rel_names, len(rel_names), self_loop=True, con="RGCN")
        self.conv2 = RelGraphConvLayer(hid_feats, hid_feats, rel_names, len(rel_names), self_loop=True, con="RGCN")
        self.conv3 = RelGraphConvLayer(in_feats, out_feats, rel_names, len(rel_names), self_loop=True, con="RGCN")

    def forward(self, graph, inputs):
        h1 = self.conv1(graph, inputs)
        # h1 = {k: F.relu(v) for k, v in h1.items()}
        h2 = self.conv2(graph, h1)
        # h2 = {k: F.relu(v) for k, v in h2.items()}
        h3 = self.conv3(graph, h2)
        h = {k: torch.cat((h1[k], h2[k], h3[k]), 1) for k, v in h1.items()}
        return h


class RGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, args):
        super().__init__()
        self.conv1 = RelGraphConvLayer(in_feats, hid_feats, rel_names, len(rel_names), self_loop=True,
                                       num_heads=args.multi_head, con="RGAT")
        self.conv2 = RelGraphConvLayer(hid_feats, hid_feats, rel_names, len(rel_names), self_loop=True,
                                       num_heads=args.multi_head, con="RGAT")
        self.conv3 = RelGraphConvLayer(in_feats, out_feats, rel_names, len(rel_names), self_loop=True,
                                       num_heads=args.multi_head, con="RGAT")

    def forward(self, graph, inputs):
        h1 = self.conv1(graph, inputs)
        # h1 = {k: F.relu(v) for k, v in h1.items()}
        h2 = self.conv2(graph, h1)
        # h2 = {k: F.relu(v) for k, v in h2.items()}
        h3 = self.conv3(graph, h2)
        h = {k: torch.cat((h1[k], h2[k], h3[k]), 1) for k, v in h1.items()}
        return h
