import numpy as np
import torch
import torch.nn as nn
import dgl
import torch as th

from tools.tools import l2_norm


class BetaAttention(nn.Module):
    def __init__(self, mods, attn_drop, ntypes, in_size: int, out_size: int):
        super(BetaAttention, self).__init__()
        self.mods = nn.ModuleDict(mods)
        self.att = nn.ParameterDict(
            {k: nn.Parameter(torch.empty(size=(1, out_size)), requires_grad=True) for k in ntypes})
        for k in ntypes:
            nn.init.xavier_uniform_(self.att[k], gain=1.414)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.fc = nn.Linear(out_size, out_size, bias=True)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {dty: {} for dty in g.dsttypes}
        beta = {dty: {} for dty in g.dsttypes}
        attn_curr = {}
        for k in g.dsttypes:
            attn_curr[k] = self.attn_drop(self.att[k])

        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            if stype not in inputs:
                continue
            dstdata = self.mods[etype](
                rel_graph,
                (inputs[stype], inputs[dtype]),
                *mod_args.get(etype, ()),
                **mod_kwargs.get(etype, {}))
            sp = self.tanh(self.fc(dstdata)).mean(dim=0)
            beta[dtype][stype] = (attn_curr.get(dtype).matmul(sp.t()))
            outputs[dtype][stype] = dstdata
        for k, v in beta.items():
            beta[k] = list(v.values())
            beta[k] = torch.cat(beta[k], dim=-1).view(-1)
            beta[k] = self.softmax(beta[k])
        for k, v in outputs.items():
            outputs[k] = list(v.values())

        rsts = {k: 0 for k in g.dsttypes}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                for i in range(len(alist)):
                    rsts[nty] += alist[i] * beta[nty][i]
        return rsts


class AlphaAttention(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 rel_names,
                 ntypes,
                 attn_drop,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.2,
                 num_heads=3):
        super(AlphaAttention, self).__init__()
        self.in_feat = in_size
        self.out_feat = out_size
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = BetaAttention({
            rel: dgl.nn.pytorch.GATConv(in_size, out_size, num_heads=num_heads, residual=True, bias=True)
            for rel in rel_names
        }, attn_drop, ntypes, in_size, out_size)

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dgl.nn.pytorch.WeightBasis((in_size, out_size), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_size, out_size))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_size))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_size, out_size))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        g = g.local_var()
        wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)
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


class ScEncoder(nn.Module):
    def __init__(self, in_feat, out_feat, rel_name, ntypes, attn_drop,
                 num_bases, weight=True, bias=True, self_loop=True, dropout=0.5, num_heads=8):
        super(ScEncoder, self).__init__()
        self.conv1 = AlphaAttention(in_feat, out_feat, rel_name, ntypes, attn_drop, num_bases, weight=weight, bias=bias,
                                    self_loop=self_loop, dropout=dropout, num_heads=num_heads)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = {k: l2_norm(v) for k, v in h.items()}
        return h
