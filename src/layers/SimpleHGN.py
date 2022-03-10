"""Heterograph NN modules"""
from functools import partial

import torch
import torch as th
import torch.nn as nn
from dgl.base import DGLError


class SimpleHGNLayer(nn.Module):

    def __init__(self, mods, aggregate='att'):
        super(SimpleHGNLayer, self).__init__()
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = self.get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

        self.w_e_drug = nn.Parameter(th.FloatTensor(5))
        self.w_e_protein = nn.Parameter(torch.Tensor(5))
        self.w_e_disease = nn.Parameter(torch.Tensor(4))
        self.w_e_sideeffect = nn.Parameter(torch.Tensor(4))
        torch.nn.init.normal_(self.w_e_drug, mean=1, std=0.5)
        torch.nn.init.normal_(self.w_e_protein, mean=1, std=0.5)
        torch.nn.init.normal_(self.w_e_disease, mean=1, std=0.5)
        torch.nn.init.normal_(self.w_e_sideeffect, mean=1, std=0.5)
        self.w_dict = {'drug': self.w_e_drug, 'protein': self.w_e_protein,
                       'disease': self.w_e_disease, 'sideeffect': self.w_e_sideeffect}

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):

        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                # if rel_graph.number_of_edges() == 0:
                #     continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                # if rel_graph.number_of_edges() == 0:
                #     continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts

    def reset_parameters(self):
        torch.nn.init.normal_(self.w_e_drug, mean=1, std=0.5)
        torch.nn.init.normal_(self.w_e_protein, mean=1, std=0.5)
        torch.nn.init.normal_(self.w_e_disease, mean=1, std=0.5)
        torch.nn.init.normal_(self.w_e_sideeffect, mean=1, std=0.5)

    def _max_reduce_func(self, inputs, dim, dtype):
        return th.max(inputs, dim=dim)[0]

    def _min_reduce_func(self, inputs, dim, dtype):
        return th.min(inputs, dim=dim)[0]

    def _sum_reduce_func(self, inputs, dim, dtype):
        return th.sum(inputs, dim=dim)

    def _mean_reduce_func(self, inputs, dim, dtype):
        return th.mean(inputs, dim=dim)

    def _attention_agg_func(self, inputs, dim, dtype):
        if len(inputs) == 0:
            return None
        res = torch.mul(inputs.T, self.w_dict[dtype]).T
        return th.mean(res, dim=dim)

    def _stack_agg_func(self, inputs, dtype):
        if len(inputs) == 0:
            return None
        return th.stack(inputs, dim=1)

    def _agg_func(self, inputs, dtype, fn):
        if len(inputs) == 0:
            return None
        stacked = th.stack(inputs, dim=0)
        return fn(stacked, dim=0, dtype=dtype)

    def get_aggregate_fn(self, agg):
        if agg == 'sum':
            fn = self._sum_reduce_func
        elif agg == 'max':
            fn = self._max_reduce_func
        elif agg == 'min':
            fn = self._min_reduce_func
        elif agg == 'mean':
            fn = self._mean_reduce_func
        elif agg == 'att':
            fn = self._attention_agg_func
        elif agg == 'stack':
            fn = None  # will not be called
        else:
            raise DGLError('Invalid cross type aggregator. Must be one of '
                           '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
        if agg == 'stack':
            return self._stack_agg_func
        else:
            return partial(self._agg_func, fn=fn)
