import dgl
import torch.nn.functional as F
import torch.nn as nn
import torch as th
from tools.tools import ConstructGraph, load_data
import numpy as np
from src.layers.SimpleHGN import myGATConv, SimpleHGNLayer


class MLPPredicator(nn.Module):
    def __init__(self, n_i, n_o):
        super(MLPPredicator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_i, int(n_i / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(n_i / 2), int(n_i / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(n_i / 4), n_o)
        )

    def froward(self, x):
        out = self.linear(x)
        return out


# 信息传递框架编程范式
# 1. 信息构建组件
# 2. 节点聚合组件
# 3. 表征更新组件

class GnnNet(nn.Module):
    def __init__(self, g, n_drug, n_protein, n_disease, n_sideeffect, args):
        super(GnnNet, self).__init__()
        self.g = g
        self.device = th.device(args.device)
        self.dim_embedding = args.dim_embedding

        self.activation = F.elu
        self.reg_lambda = args.reg_lambda

        self.num_disease = n_disease
        self.num_drug = n_drug
        self.num_protein = n_protein
        self.num_sideeffect = n_sideeffect

        # load feature
        self.drug_feat = th.from_numpy(np.loadtxt("../../data/feature/drug_feature.txt")).to(th.float32)
        nn.init.normal_(self.drug_feat, mean=0, std=0.1)
        self.protein_feat = th.from_numpy(np.loadtxt("../../data/feature/protein_feature.txt")).to(th.float32)
        nn.init.normal_(self.protein_feat, mean=0, std=0.1)
        self.disease_feat = th.from_numpy(np.loadtxt("../../data/feature/disease_feature.txt")).to(th.float32)
        nn.init.normal_(self.disease_feat, mean=0, std=0.1)
        self.sideeffect_feat = th.from_numpy(np.loadtxt("../../data/feature/sideeffect_feature.txt")).to(th.float32)
        nn.init.normal_(self.sideeffect_feat, mean=0, std=0.1)

        # 定义网络结构

        self.layer1 = SimpleHGNLayer()
        self.layer2 = SimpleHGNLayer()
        self.dropout = args.dropout

    def forward(self):
        h = self.layer1()
        h = self.layer1()
        return h


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sage = GnnNet()
        self.pred = MLPPredicator(128, 2)

    def forward(self, g, neg_samples, x, etype: list):
        h = self.sage()

        return self.pred(), self.pred()
