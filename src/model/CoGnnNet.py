import torch.nn as nn
import torch.nn.functional as F
import dgl
from src.layers.mp_encoder import MpEncoder
from src.layers.sc_encoder import ScEncoder
from src.layers.contrast import Contrast


class HeCo(nn.Module):
    def __init__(self, in_size, out_size, mps_dict: dict, g,
                 attn_drop, num_bases, tau, lam, keys):
        super(HeCo, self).__init__()
        self.feat = nn.Linear(in_size, out_size)
        self.mp = MpEncoder(out_size, out_size, mps_dict)
        self.sc = ScEncoder(out_size, out_size, g.etypes, g.ntypes, attn_drop, num_bases)
        self.contrast = Contrast(out_size, tau, lam, keys)

    def get_mp_embeds(self, mp_graph_dict, h, mps_key, mps):
        h = {k: self.feat(v) for k, v in h.items()}
        return self.mp(mp_graph_dict, h, mps_key, mps)

    def get_sc_embeds(self, graph, h):
        h = {k: self.feat(v) for k, v in h.items()}
        return self.sc(graph, h)

    def forward(self, graph, meta_graph_dict, h, pos_dict, mps_key_dict, mps_dict):
        h = {k: self.feat(v) for k, v in h.items()}
        z_mp = self.mp(meta_graph_dict, h, mps_key_dict, mps_dict)
        z_sc = self.sc(graph, h)
        loss = self.contrast(z_mp, z_sc, pos_dict)
        return loss
