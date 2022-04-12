import dgl.nn.pytorch
import torch
import torch.nn as nn
import torch as th

from tools.tools import l2_norm


class BetaAttention(nn.Module):
    def __init__(self, mods, meta_paths: list, attn_drop: float, in_size: int, out_size: int):
        super(BetaAttention, self).__init__()
        self.mods = nn.ModuleDict(mods)
        self.att = nn.ParameterDict(
            {k: nn.Parameter(torch.empty(size=(1, out_size)), requires_grad=True) for k in meta_paths})
        for k in meta_paths:
            nn.init.xavier_uniform_(self.att[k], gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.fc = nn.Linear(out_size, out_size, bias=True)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, g, inputs, meta_path_key, meta_paths, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = []
        beta = []
        attn_curr = {}
        for meta_path in meta_path_key:
            attn_curr[meta_path] = self.attn_drop(self.att[meta_path])

        for i, meta_path in enumerate(meta_paths):
            new_g = g[i]
            new_g = new_g.local_var()
            dstdata = self.mods[meta_path_key[i]](new_g, inputs)
            sp = self.tanh(self.fc(dstdata)).mean(dim=0)
            beta.append(attn_curr.get(meta_path_key[i]).matmul(sp.t()))
            outputs.append(dstdata)

        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)

        rsts = 0
        for i, output in enumerate(outputs):
            rsts += output * beta[i]
        return rsts


class Attention(nn.Module):
    def __init__(self, in_size, out_size, meta_paths, attn_drop: float,
                 weight=True, bias=True, activation=None, self_loop=False, dropout=0.2):
        super(Attention, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.meta_paths = meta_paths
        self.attn_drop = attn_drop
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.conv = BetaAttention({
            mp: dgl.nn.pytorch.GraphConv(in_size, out_size) for mp in meta_paths
        }, meta_paths, attn_drop, in_size, out_size)

        self.use_weight = weight

        num_bases = len(meta_paths)
        self.use_bias = num_bases < len(meta_paths) and weight
        if self.use_weight:
            if self.use_bias:
                self.bias = dgl.nn.pytorch.WeightBasis((in_size, out_size), num_bases, len(self.meta_paths))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.meta_paths), in_size, out_size))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_size))
            nn.init.zeros_(self.h_bias)

        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_size, out_size))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs, meta_path_key, meta_paths):
        if self.use_weight:
            weight = self.bias() if self.use_bias else self.weight
            wdict = {self.meta_paths[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}

        inputs_dst = inputs

        hs = self.conv(g, inputs, meta_path_key, meta_paths, mod_kwargs=wdict)

        def _apply(h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst, self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return _apply(hs)


class MpEncoder(nn.Module):
    def __init__(self, in_size, out_size, mps_dict: dict):
        super(MpEncoder, self).__init__()
        self.conv = nn.ModuleDict({k: Attention(in_size, out_size, v, attn_drop=0.2) for k, v in mps_dict.items()})

    def forward(self, g, h, mps_key_dict: dict, mps_dict: dict):
        embeds = {}
        for k, v in mps_key_dict.items():
            node_feature = self.conv[k](g[k], h[k], v, mps_dict[k])
            embeds[k] = l2_norm(node_feature)
        return embeds
