import dgl
import torch
import torch.nn as nn
from src.layers.RGCNLayer import RGCN, RGAT
from src.layers.MLPPredicator import MLPPredicator
from src.layers.DistMulLayer import DistLayer
from src.tools.tools import shuffle, predict_target_pair, get_meta_path, l2_norm


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

        # self.pred = MLPPredicator(out_features * 3 * 2, 1)
        self.pred = DistLayer(out_features)

    def forward(self, g, neg_g, x, etype, pos_g=None, drug_drug=None, drug_chemical=None, protein_protein=None,
                protein_sequence=None, drug_protein=None, mask=None):

        h = self.sage(g, x)
        for k, v in h.items():
            h[k] = l2_norm(v)
        return self.pred(drug_drug, drug_chemical, protein_protein, protein_sequence,
                         drug_protein, mask, h['drug'], h['protein'])

        # def concat_message_function(edges):
        #     return {
        #         'cat_feat': torch.cat([edges.src['feature'], edges.dst['feature']],
        #                               len(edges.src['feature'].shape) - 1)}
        #
        # def concat_feature(g):
        #     with g.local_scope():
        #         g.nodes['drug'].data['feature'] = h['drug']
        #         g.nodes['protein'].data['feature'] = h['protein']
        #         g.apply_edges(concat_message_function, etype=etype)
        #         return g.edata.pop('cat_feat')
        #
        # pos_h, neg_h = concat_feature(g), concat_feature(neg_g)
        # if isinstance(pos_h, dict):
        #     pos_h = pos_h[etype]
        # if isinstance(neg_h, dict):
        #     neg_h = neg_h[etype]
        #
        # # pre, target = shuffle(pos_h, neg_h)
        # #
        # # return self.pred(pre), target, h
        #
        # pre, target = predict_target_pair(self.pred(pos_h), self.pred(neg_h))
        #
        # return pre, target, h
