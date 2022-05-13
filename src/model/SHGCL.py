import time

import dgl
import torch.nn.functional as F
import torch as th
import torch.nn as nn

from layers.DistMult import DistMult
from tools.tools import row_normalize, l2_norm
from layers.contrast import Contrast
from layers.mp_encoder import MpEncoder

drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'


class SemanticsAttention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(SemanticsAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(th.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = th.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp


class SCencoder(nn.Module):
    def __init__(self, out_dim, keys):
        super(SCencoder, self).__init__()
        self.dim_embedding = out_dim
        self.keys = keys
        self.fc_DDI = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_ch = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_Di = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_Side = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_PPI = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_seq = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_Di = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_P_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Di_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Di_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_Side_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        # self.semantics_attention = {k: SemanticsAttention(out_dim, attn_drop=0.2) for k in keys}
        self.reset_parameters()

    def reset_parameters(self):
        for m in SCencoder.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                protein_sequence, protein_disease, drug_protein, node_feature: dict) -> dict:
        disease_feature = node_feature[disease]
        drug_feature = node_feature[drug]
        protein_feature = node_feature[protein]
        sideeffect_feature = node_feature[sideeffect]
        node_feat = {}
        disease_agg = [th.mm(row_normalize(drug_disease.T).float(), F.relu(self.fc_Di_D(drug_feature))),
                       th.mm(row_normalize(protein_disease.T).float(), F.relu(self.fc_Di_P(protein_feature))),
                       disease_feature]
        drug_agg = [th.mm(row_normalize(drug_drug).float(), F.relu(self.fc_DDI(drug_feature))),
                    th.mm(row_normalize(drug_chemical).float(), F.relu(self.fc_D_ch(drug_feature))),
                    th.mm(row_normalize(drug_disease).float(), F.relu(self.fc_D_Di(disease_feature))),
                    th.mm(row_normalize(drug_sideeffect).float(), F.relu(self.fc_D_Side(sideeffect_feature))),
                    th.mm(row_normalize(drug_protein).float(), F.relu(self.fc_D_P(protein_feature))),
                    drug_feature]
        protein_agg = [th.mm(row_normalize(protein_protein).float(), F.relu(self.fc_PPI(protein_feature))),
                       th.mm(row_normalize(protein_sequence).float(), F.relu(self.fc_P_seq(protein_feature))),
                       th.mm(row_normalize(protein_disease).float(), F.relu(self.fc_P_Di(disease_feature))),
                       th.mm(row_normalize(drug_protein.T).float(), F.relu(self.fc_P_D(drug_feature))),
                       protein_feature]
        sideeffect_agg = [th.mm(row_normalize(drug_sideeffect.T).float(), F.relu(self.fc_Side_D(drug_feature))),
                          sideeffect_feature]
        # agg_dict = {drug: drug_agg, protein: protein_agg, disease: disease_agg, sideeffect: sideeffect_agg}
        # for k, v in agg_dict.items():
        #     node_feat[k] = self.semantics_attention[k](agg_dict[k])
        disease_feat = th.mean(th.stack(disease_agg, dim=1), dim=1)
        drug_feat = th.mean(th.stack(drug_agg, dim=1), dim=1)
        protein_feat = th.mean(th.stack(protein_agg, dim=1), dim=1)
        sideeffect_feat = th.mean(th.stack(sideeffect_agg, dim=1), dim=1)
        node_feat = {'drug': drug_feat, 'protein': protein_feat, 'sideeffect': sideeffect_feat, 'disease': disease_feat}
        return node_feat


class SHGCL(nn.Module):
    def __init__(self, node_feature, out_dim, args, keys, mps_len_dict: dict, attn_drop):
        super(SHGCL, self).__init__()
        self.device = th.device(args.device)
        self.dim_embedding = out_dim
        self.keys = keys
        self.reg_lambda = args.reg_lambda

        self.fc_dict = nn.ModuleDict({k: nn.Linear(128, out_dim) for k in keys})
        self.scencoder = SCencoder(out_dim, keys)
        # self.scencoder1 = SCencoder(out_dim, keys)
        self.mpencoders = nn.ModuleDict({k: MpEncoder(v, out_dim, attn_drop) for k, v in mps_len_dict.items()})
        self.constrast = Contrast(out_dim, args.tau, args.lam, keys)
        self.distmult = DistMult(self.dim_embedding)
        self.reset_parameters()

    def reset_parameters(self):
        for m in SHGCL.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                protein_sequence, protein_disease, drug_protein, drug_protein_mask,
                mps_dict: dict, pos_dict: dict, cl, node_feature: dict):
        node_f = {k: self.fc_dict[k](node_feature[k]) for k, v in node_feature.items()}
        # node_f = {drug: self.drug_feat, protein: self.protein_feat,
        #           sideeffect: self.sideeffect_feat, disease: self.disease_feat}
        node_sc = self.scencoder(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                 protein_sequence, protein_disease, drug_protein, node_f)
        # node_sc = self.scencoder1(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
        #                          protein_sequence, protein_disease, drug_protein, node_sc)
        node_mp = {k: self.mpencoders[k](node_f[k], mps_dict[k]) for k, v in mps_dict.items()}
        cl_loss = self.constrast(node_sc, node_mp, pos_dict)
        # node_act = {k: th.cat((node_sc[k], node_mp[k]), 1) for k in self.keys}
        node_act = node_sc
        disease_vector = l2_norm(node_act[disease])
        drug_vector = l2_norm(node_act[drug])
        protein_vector = l2_norm(node_act[protein])
        sideeffect_vector = l2_norm(node_act[sideeffect])

        mloss, dti_re = self.distmult(drug_vector, disease_vector, sideeffect_vector, protein_vector,
                                      drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                      protein_sequence, protein_disease, drug_protein, drug_protein_mask)

        loss = mloss + cl * cl_loss
        return loss, dti_re.detach()
