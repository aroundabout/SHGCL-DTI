import torch
import torch.nn as nn
from src.layers.RGCN import RGCN
from src.layers.RGAT import RGAT
from src.layers.MLPPredicator import MLPPredicator


class GnnModel(nn.Module):
    def __init__(self, g, in_features, hidden_features, out_features, args):
        super(GnnModel, self).__init__()
        self.g = g

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        if args.model == 'RGAT':
            self.sage = RGAT(in_features, hidden_features, out_features, g.etypes, args)
        else:
            self.sage = RGCN(in_features, hidden_features, out_features, g.etypes, args)
        self.pred = MLPPredicator(out_features * 2, 1)

    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)

        def concat_message_function(edges):
            return {
                'cat_feat': torch.cat([edges.src['feature'], edges.dst['feature']],len(edges.src['feature'].shape)-1)}

        with g.local_scope():
            g.nodes['drug'].data['feature'] = h['drug']
            g.nodes['protein'].data['feature'] = h['protein']
            g.apply_edges(concat_message_function, etype=etype)
            pos_h = g.edata.pop('cat_feat')
        with neg_g.local_scope():
            neg_g.nodes['drug'].data['feature'] = h['drug']
            neg_g.nodes['protein'].data['feature'] = h['protein']
            neg_g.apply_edges(concat_message_function, etype=etype)
            neg_h = neg_g.edata.pop('cat_feat')
        if isinstance(pos_h, dict):
            pos_h = pos_h[etype]
        if isinstance(neg_h, dict):
            neg_h = neg_h[etype]
        return self.pred(pos_h), self.pred(neg_h), h
