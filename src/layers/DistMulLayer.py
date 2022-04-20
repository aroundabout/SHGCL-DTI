import torch.nn as nn
import torch as th
import numpy as np

from src.tools.tools import predict_target_pair, compute_score


class DistLayer(nn.Module):
    def __init__(self, dim_embedding):
        super(DistLayer, self).__init__()
        self.dim_embedding = dim_embedding
        tmp = th.randn(self.dim_embedding).float()
        self.re_DTI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1))).float()
        self.re_DR_CH = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1))).float()
        self.re_DR_DR = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1))).float()
        self.re_PR_PR = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1))).float()
        self.re_PR_SEQ = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1))).float()

        # 非durg和protein关系
        self.re_DR_DI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1))).float()
        self.re_DR_SE = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1))).float()
        self.re_PR_DI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1))).float()

    def forward(self, drug_drug, drug_chemical, protein_protein, protein_sequence,
                drug_protein, drug_disease, drug_sideeffect, protein_disease, mask, node_feature):
        drug_feature = node_feature['drug']
        protein_feature = node_feature['protein']
        sideeffect_feature = node_feature['sideeffect']
        disease_feature = node_feature['disease']
        drug_protein_reconstruct = th.mm(th.mm(drug_feature, self.re_DTI), protein_feature.t())
        drug_drug_reconstruct = th.mm(th.mm(drug_feature, self.re_DR_DR), drug_feature.t())
        drug_ch_reconstruct = th.mm(th.mm(drug_feature, self.re_DR_CH), drug_feature.t())
        protein_protein_reconstruct = th.mm(th.mm(protein_feature, self.re_PR_PR), protein_feature.t())
        protein_seq_reconstruct = th.mm(th.mm(protein_feature, self.re_PR_SEQ), protein_feature.t())

        # 非drug和protein
        drug_disease_reconstruct = th.mm(th.mm(drug_feature, self.re_DR_DI), disease_feature.t())
        drug_sideeffect_reconstruct = th.mm(th.mm(drug_feature, self.re_DR_SE), sideeffect_feature.t())
        protein_disease_reconstruct = th.mm(th.mm(protein_feature, self.re_PR_DI), disease_feature.t())
        DTI_potential = drug_protein_reconstruct - drug_protein.float()

        # drug_protein_reconstruct_loss = th.sum((drug_protein_reconstruct - drug_protein.float()) ** 2)
        tmp = th.mul(mask.float(), (drug_protein_reconstruct - drug_protein.float()))
        drug_protein_reconstruct_loss = th.sum(tmp ** 2)
        drug_drug_reconstruct_loss = th.sum((drug_drug_reconstruct - drug_drug.float()) ** 2)
        drug_ch_reconstruct_loss = th.sum((drug_ch_reconstruct - drug_chemical.float()) ** 2)
        protein_protein_reconstruct_loss = th.sum((protein_protein_reconstruct - protein_protein.float()) ** 2)
        protein_seq_reconstruct_loss = th.sum((protein_seq_reconstruct - protein_sequence.float()) ** 2)

        # 非durg和protein
        drug_disease_loss = th.sum((drug_disease_reconstruct - drug_disease.float()) ** 2)
        drug_sideeffect_loss = th.sum((drug_sideeffect_reconstruct - drug_sideeffect.float()) ** 2)
        protein_disease_loss = th.sum((protein_disease_reconstruct - protein_disease.float()) ** 2)


        other_loss = drug_drug_reconstruct_loss + drug_ch_reconstruct_loss + \
                     protein_protein_reconstruct_loss + protein_seq_reconstruct_loss
        other_loss = drug_drug_reconstruct_loss + drug_ch_reconstruct_loss + \
                     protein_protein_reconstruct_loss + protein_seq_reconstruct_loss + \
                     drug_disease_loss + drug_sideeffect_loss + protein_disease_loss
        tloss = drug_protein_reconstruct_loss + 1.0 * other_loss
        # tloss = drug_protein_reconstruct_loss

        return tloss, drug_protein_reconstruct
