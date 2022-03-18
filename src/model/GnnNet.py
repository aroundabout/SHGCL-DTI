import dgl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch as th
import numpy as np

from tools.tools import ConstructGraph, load_data
from src.layers.SimpleHGN import SimpleHGNLayer
from src.layers.RGCN import RGCN
from src.layers.MLPPredicator import MLPPredicator


class GnnModel(nn.Module):
    def __init__(self, g, in_features, hidden_features, out_features, args):
        super(GnnModel, self).__init__()
        self.g = g
        # self.device = th.device(args.device)
        # self.dim_embedding = args.dim_embedding
        #
        # self.activation = F.elu
        # self.reg_lambda = args.reg_lambda

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        # self.num_disease = n_disease
        # self.num_drug = n_drug
        # self.num_protein = n_protein
        # self.num_sideeffect = n_sideeffect
        #
        # # load feature
        # self.drug_feat = th.from_numpy(np.loadtxt("../../data/feature/drug_feature.txt")).to(th.float32)
        # nn.init.normal_(self.drug_feat, mean=0, std=0.1)
        # self.protein_feat = th.from_numpy(np.loadtxt("../../data/feature/protein_feature.txt")).to(th.float32)
        # nn.init.normal_(self.protein_feat, mean=0, std=0.1)
        # self.disease_feat = th.from_numpy(np.loadtxt("../../data/feature/disease_feature.txt")).to(th.float32)
        # nn.init.normal_(self.disease_feat, mean=0, std=0.1)
        # self.sideeffect_feat = th.from_numpy(np.loadtxt("../../data/feature/sideeffect_feature.txt")).to(th.float32)
        # nn.init.normal_(self.sideeffect_feat, mean=0, std=0.1)

        # 定义网络结构
        # if args.
        self.sage = RGCN(in_features, hidden_features, out_features, g.etypes)
        self.pred = MLPPredicator(out_features * 2, 1)
        # self.dropout = args.dropout

    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)

        def concat_message_function(edges):
            return {'cat_feat': torch.cat([edges.src['feature'], edges.dst['feature']], 1)}

        with g.local_scope():
            g.apply_edges(concat_message_function, etype=etype)
            pos_h = g.edata.pop('cat_feat')
        with neg_g.local_scope():
            neg_g.apply_edges(concat_message_function, etype=etype)
            neg_h = neg_g.edata.pop('cat_feat')
        if isinstance(pos_h, dict):
            pos_h = pos_h[etype]
        if isinstance(neg_h, dict):
            neg_h = neg_h[etype]
        return self.pred(pos_h), self.pred(neg_h), h
