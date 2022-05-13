import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../')
from layers.mp_encoder import MpEncoder
from layers.sc_encoder import ScEncoder
from layers.contrast import Contrast
from layers.MLPPredicator import MLPPredicator
from tools.tools import concat_link


class HSGL(nn.Module):
    def __init__(self, in_size, out_size, mps_dict: dict, g,
                 attn_drop, num_bases, tau, lam, keys, feat_drop,
                 feat_size_dict=None):
        super(HSGL, self).__init__()
        if feat_size_dict is None:
            feat_size_dict = {'drug': in_size, 'protein': in_size, 'sideeffect': in_size, 'disease': in_size}
        self.fc_list = nn.ModuleDict({k: nn.Linear(v, out_size) for k, v in feat_size_dict.items()})
        for k, v in self.fc_list.items():
            nn.init.xavier_normal_(v.weight, gain=1.414)
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.mp = nn.ModuleDict({k: MpEncoder(v, out_size, attn_drop) for k, v in mps_dict.items()})
        self.mp2 = nn.ModuleDict({k: MpEncoder(v, out_size, attn_drop) for k, v in mps_dict.items()})
        self.sc = ScEncoder(out_size, out_size, g.etypes, g.ntypes, attn_drop, num_bases)
        self.sc2 = ScEncoder(out_size, out_size, g.etypes, g.ntypes, attn_drop, num_bases)
        self.contrast = Contrast(out_size, tau, lam, keys)
        self.pred = MLPPredicator(out_size * 4, 1)

    def get_mp_embeds(self, h, mps_dict):
        h_all = {}
        for k, v in mps_dict.items():
            h_all[k] = F.elu(self.fc_list[k](h[k]))
            h_all[k] = self.mp[k](h_all[k], mps_dict[k])
        return h_all

    def get_sc_embeds(self, graph, h):
        h_all = {}
        for k, v in h.items():
            h_all[k] = F.elu(self.fc_list[k](h[k]))
        return self.sc(graph, h_all)

    def get_out_h(self, graph, mps_dict: dict, h, dti):
        h_all = {}
        z_mp = {}
        z = {}
        for k, v in h.items():
            h_all[k] = F.elu(self.feat_drop(self.fc_list[k](h[k])))
        z_sc = self.sc(graph, h_all)
        for k, v in mps_dict.items():
            z_mp[k] = self.mp[k](h_all[k], mps_dict[k])
        for k in mps_dict.keys():
            z[k] = torch.cat((z_sc[k], z_mp[k]), 1)
        out_h = self.pred(concat_link(dti, z['drug'], z['protein']))
        return out_h

    def forward(self, graph, mps_dict: dict, h, pos_dict, dti):
        h_all = {}
        z_mp = {}
        for k, v in h.items():
            h_all[k] = F.elu(self.feat_drop(self.fc_list[k](h[k])))
        z_sc = self.sc(graph, h_all)
        z_sc = self.sc2(graph, z_sc)

        for k, v in mps_dict.items():
            z_mp[k] = self.mp[k](h_all[k], mps_dict[k])
        for k, v in mps_dict.items():
            z_mp[k] = self.mp2[k](z_mp[k], mps_dict[k])
        loss = self.contrast(z_mp, z_sc, pos_dict)
        z = {}
        for k in mps_dict.keys():
            z[k] = torch.cat((z_sc[k], z_mp[k]), 1)
        in_h = concat_link(dti, z['drug'], z['protein'])
        out_h = self.pred(in_h)
        return loss, out_h
