import dgl
import torch
import torch.nn.functional as F
import torch as th
import torch.nn as nn
from src.tools.tools import l2_norm, row_normalize

from src.tools.tools import construct_postive_graph, construct_negative_graph, predict_target_pair, compute_score, \
    l2_norm
from src.layers.MLPPredicator import MLPPredicator
from src.layers.DistMulLayer import DistLayer

drug = 'drug'
protein = 'protein'
sideeffect = 'sideeffect'
disease = 'disease'


class DTIGCN(nn.Module):
    def __init__(self, edge_dict: dict, out_size: int):
        super(DTIGCN, self).__init__()
        self.drug_moduleDict = nn.ModuleDict({k: nn.Linear(out_size, out_size) for k in edge_dict[drug]})
        self.protein_moduleDict = nn.ModuleDict({k: nn.Linear(out_size, out_size) for k in edge_dict[protein]})
        self.disease_moduleDict = nn.ModuleDict({k: nn.Linear(out_size, out_size) for k in edge_dict[disease]})
        self.sideeffect_moduleDict = nn.ModuleDict({k: nn.Linear(out_size, out_size) for k in edge_dict[sideeffect]})

    def forward(self, node_features: dict, edge_dict: dict, feat_dict: dict):
        drug_feat = node_features[drug]
        protein_feat = node_features[protein]
        disease_feat = node_features[disease]
        sideeffect_feat = node_features[sideeffect]
        drug_feat_list = []
        protein_feat_list = []
        sideeffect_feat_list = []
        disease_feat_list = []
        feat_dict_new = {}
        for k, v in feat_dict.items():
            temp = {}
            for k1, v1 in v.items():
                temp[k1] = node_features[v1]
            feat_dict_new[k] = temp
        # 这里的v是drug_drug drug_protein等
        for k, v in edge_dict[drug].items():
            drug_feat_list.append(
                th.mm(row_normalize(v).float(), F.relu(self.drug_moduleDict[k](feat_dict_new[drug][k]))))
        drug_feat_list.append(drug_feat)
        drug_feat_agg = l2_norm(th.mean(th.stack(drug_feat_list, dim=1), dim=1))

        for k, v in edge_dict[protein].items():
            protein_feat_list.append(
                th.mm(row_normalize(v).float(), F.elu(self.protein_moduleDict[k](feat_dict_new[protein][k]))))
        protein_feat_list.append(protein_feat)
        protein_feat_agg = l2_norm(th.mean(th.stack(protein_feat_list, dim=1), dim=1))

        for k, v in edge_dict[sideeffect].items():
            sideeffect_feat_list.append(
                th.mm(row_normalize(v).float(), F.elu(self.sideeffect_moduleDict[k](feat_dict_new[sideeffect][k]))))
        sideeffect_feat_list.append(sideeffect_feat)
        sideeffect_feat_agg = l2_norm(th.mean(th.stack(sideeffect_feat_list, dim=1), dim=1))

        for k, v in edge_dict[disease].items():
            disease_feat_list.append(
                th.mm(row_normalize(v).float(), F.elu(self.disease_moduleDict[k](feat_dict_new[disease][k]))))
        disease_feat_list.append(disease_feat)
        disease_feat_agg = l2_norm(th.mean(th.stack(disease_feat_list, dim=1), dim=1))

        feat = {drug: drug_feat_agg, protein: protein_feat_agg, sideeffect: sideeffect_feat_agg,
                disease: disease_feat_agg}
        return feat


class GnnNetV2(nn.Module):
    def __init__(self, out_size, feat_drop, edge_dict, feat_size_dict=None):
        super().__init__()
        if feat_size_dict is None:
            feat_size_dict = {'drug': 128, 'protein': 128, 'sideeffect': 128, 'disease': 128}
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.fc_list = nn.ModuleDict({k: nn.Linear(v, out_size) for k, v in feat_size_dict.items()})
        # self.conv1 = DTIGCN(edge_dict, out_size)
        # self.conv2 = DTIGCN(edge_dict, out_size)
        self.decoder = DistLayer(out_size)

    def forward(self, h, edge_dict: dict, feat_dict: dict, drug_drug, drug_chemical, protein_protein,
                protein_sequence, drug_protein, drug_disease, drug_sideeffect, protein_disease, mask):
        h_all = {}
        for k, v in h.items():
            # h_all[k] = self.feat_drop(self.fc_list[k](h[k]))
            h_all[k] = self.fc_list[k](h[k])
        # h_all = self.conv1(h_all, edge_dict, feat_dict)
        # h_all = self.conv2(h_all, edge_dict, feat_dict)
        tloss, dti_reconstruct = self.decoder(drug_drug, drug_chemical, protein_protein, protein_sequence,
                                              drug_protein, drug_disease, drug_sideeffect, protein_disease, mask,
                                              h_all)
        return tloss, dti_reconstruct


class GRDTI(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GRDTI, self).__init__()
        self.dim_embedding = out_dim
        feat_size_dict = {'drug': in_dim, 'protein': in_dim, 'sideeffect': in_dim, 'disease': in_dim}
        self.fc_list = nn.ModuleDict({k: nn.Linear(v, out_dim) for k, v in feat_size_dict.items()})

        # 邻居信息的权重矩阵，对应论文公式（1）中的Wr、br
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

        # Linear transformation for reconstruction
        tmp = th.randn(self.dim_embedding).float()
        self.re_DDI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_ch = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_Di = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_Side = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_P = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_PPI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_P_seq = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_P_Di = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))

        self.reset_parameters()

    def reset_parameters(self):
        for m in GRDTI.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                protein_sequence, protein_disease, drug_protein, drug_protein_mask, node_feat: dict):
        node_feature = {}
        for k, v in node_feat.items():
            node_feature[k] = self.fc_list[k](node_feat[k])
            # node_feature[k] = node_feature[k].normal_(mean=0, std=0.1)
        disease_feat = th.mean(th.stack((th.mm(row_normalize(drug_disease.T).float(),
                                               F.relu(self.fc_Di_D(node_feature[drug]))),
                                         th.mm(row_normalize(protein_disease.T).float(),
                                               F.relu(self.fc_Di_P(node_feature[protein]))),
                                         node_feature[disease]), dim=1), dim=1)

        drug_feat = th.mean(th.stack((th.mm(row_normalize(drug_drug).float(),
                                            F.relu(self.fc_DDI(node_feature[drug]))),
                                      th.mm(row_normalize(drug_chemical).float(),
                                            F.relu(self.fc_D_ch(node_feature[drug]))),
                                      th.mm(row_normalize(drug_disease).float(),
                                            F.relu(self.fc_D_Di(node_feature[disease]))),
                                      th.mm(row_normalize(drug_sideeffect).float(),
                                            F.relu(self.fc_D_Side(node_feature[sideeffect]))),
                                      th.mm(row_normalize(drug_protein).float(),
                                            F.relu(self.fc_D_P(node_feature[protein]))),
                                      node_feature[drug]), dim=1), dim=1)

        protein_feat = th.mean(th.stack((th.mm(row_normalize(protein_protein).float(),
                                               F.relu(self.fc_PPI(node_feature[protein]))),
                                         th.mm(row_normalize(protein_sequence).float(),
                                               F.relu(self.fc_P_seq(node_feature[protein]))),
                                         th.mm(row_normalize(protein_disease).float(),
                                               F.relu(self.fc_P_Di(node_feature[disease]))),
                                         th.mm(row_normalize(drug_protein.T).float(),
                                               F.relu(self.fc_P_D(node_feature[drug]))),
                                         node_feature[protein]), dim=1), dim=1)

        sideeffect_feat = th.mean(th.stack((th.mm(row_normalize(drug_sideeffect.T).float(),
                                                  F.relu(self.fc_Side_D(node_feature[drug]))),
                                            node_feature[sideeffect]), dim=1), dim=1)

        disease_embedding = disease_feat
        drug_embedding = drug_feat
        protein_embedding = protein_feat
        sideeffect_embedding = sideeffect_feat

        disease_vector = l2_norm(disease_embedding)
        drug_vector = l2_norm(drug_embedding)
        protein_vector = l2_norm(protein_embedding)
        sideeffect_vector = l2_norm(sideeffect_embedding)

        drug_drug_reconstruct = th.mm(th.mm(drug_vector, self.re_DDI), drug_vector.t())
        drug_drug_reconstruct_loss = th.sum(
            (drug_drug_reconstruct - drug_drug.float()) ** 2)

        drug_chemical_reconstruct = th.mm(th.mm(drug_vector, self.re_D_ch), drug_vector.t())
        drug_chemical_reconstruct_loss = th.sum(
            (drug_chemical_reconstruct - drug_chemical.float()) ** 2)

        drug_disease_reconstruct = th.mm(th.mm(drug_vector, self.re_D_Di), disease_vector.t())
        drug_disease_reconstruct_loss = th.sum(
            (drug_disease_reconstruct - drug_disease.float()) ** 2)

        drug_sideeffect_reconstruct = th.mm(th.mm(drug_vector, self.re_D_Side), sideeffect_vector.t())
        drug_sideeffect_reconstruct_loss = th.sum(
            (drug_sideeffect_reconstruct - drug_sideeffect.float()) ** 2)

        protein_protein_reconstruct = th.mm(th.mm(protein_vector, self.re_PPI), protein_vector.t())
        protein_protein_reconstruct_loss = th.sum(
            (protein_protein_reconstruct - protein_protein.float()) ** 2)

        protein_sequence_reconstruct = th.mm(th.mm(protein_vector, self.re_P_seq), protein_vector.t())
        protein_sequence_reconstruct_loss = th.sum(
            (protein_sequence_reconstruct - protein_sequence.float()) ** 2)

        protein_disease_reconstruct = th.mm(th.mm(protein_vector, self.re_P_Di), disease_vector.t())
        protein_disease_reconstruct_loss = th.sum(
            (protein_disease_reconstruct - protein_disease.float()) ** 2)

        drug_protein_reconstruct = th.mm(th.mm(drug_vector, self.re_D_P), protein_vector.t())
        tmp = th.mul(drug_protein_mask.float(), (drug_protein_reconstruct - drug_protein.float()))
        drug_protein_reconstruct_loss = th.sum(tmp ** 2)

        other_loss = drug_drug_reconstruct_loss + drug_chemical_reconstruct_loss + drug_disease_reconstruct_loss + \
                     drug_sideeffect_reconstruct_loss + protein_protein_reconstruct_loss + \
                     protein_sequence_reconstruct_loss + protein_disease_reconstruct_loss

        # other_loss = drug_drug_reconstruct_loss + drug_chemical_reconstruct_loss + \
        #              protein_protein_reconstruct_loss + protein_sequence_reconstruct_loss

        # L2_loss = 0.
        # for name, param in GRDTI.named_parameters(self):
        #     if 'bias' not in name:
        #         L2_loss = L2_loss + th.sum(param.pow(2))
        # L2_loss = L2_loss * 0.5

        # tloss = drug_protein_reconstruct_loss + 1.0 * other_loss + self.reg_lambda * L2_loss
        tloss = drug_protein_reconstruct_loss + 1.0 * other_loss

        return tloss, drug_protein_reconstruct
