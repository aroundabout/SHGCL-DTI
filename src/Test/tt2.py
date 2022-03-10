import dgl
import dgl.nn as dglnn
import dgl.function as fn
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from dgl.nn.pytorch.conv.gatconv import GATConv
from src.layers.SimpleHGN import SimpleHGNLayer
from tools.tools import ConstructGraph, load_data, ConstructGraphWithRW

drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, protein_disease, dti_original = load_data()

# 构建异质图
hetero_graph = ConstructGraphWithRW(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                              protein_sequence,
                              protein_disease, dti_original)
hetero_graph.nodes['disease'].data['feature'] = torch.from_numpy(
    numpy.loadtxt("../../data/feature/disease_feature.txt")).to(torch.float32)
hetero_graph.nodes['drug'].data['feature'] = torch.from_numpy(
    numpy.loadtxt("../../data/feature/drug_feature.txt")).to(torch.float32)
hetero_graph.nodes['protein'].data['feature'] = torch.from_numpy(
    numpy.loadtxt("../../data/feature/protein_feature.txt")).to(torch.float32)
hetero_graph.nodes['sideeffect'].data['feature'] = torch.from_numpy(
    numpy.loadtxt("../../data/feature/sideeffect_feature.txt")).to(torch.float32)


# randomly generate training masks on user nodes and click edges


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = SimpleHGNLayer({
            rel: GATConv(in_feats, hid_feats, 3)
            for rel in rel_names}, aggregate='att')
        self.conv2 = SimpleHGNLayer({
            rel: GATConv(hid_feats, out_feats, 1)
            for rel in rel_names}, aggregate='att')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        print("foward")
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


k = 5
model = Model(128, 256, 128, hetero_graph.etypes)
drug_feats = hetero_graph.nodes['drug'].data['feature']
protein_feats = hetero_graph.nodes['protein'].data['feature']
disease_feats = hetero_graph.nodes['disease'].data['feature']
sideeffect_feats = hetero_graph.nodes['sideeffect'].data['feature']
node_features = {'drug': drug_feats, 'protein': protein_feats, 'disease': disease_feats, 'sideeffect': sideeffect_feats}
opt = torch.optim.Adam(model.parameters())
print("train start")
for epoch in range(3):
    print("now is the ", epoch, " epoch")
    negative_graph = construct_negative_graph(hetero_graph, k, ('drug', 'drug_protein interaction', 'protein'))
    print("negative graph")
    pos_score, neg_score = \
        model(hetero_graph, negative_graph, node_features, ('drug', 'drug_protein interaction', 'protein'))
    loss = compute_loss(pos_score, neg_score)
    print("loss")
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())

node_embeddings = model.sage(hetero_graph, node_features)
print(node_embeddings)
