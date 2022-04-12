import torch.nn as nn
import torch as th
import numpy as np

from src.tools.tools import predict_target_pair, compute_score


class DistLayer(nn.Module):
    def __init__(self, dim_embedding):
        super(DistLayer, self).__init__()
        self.dim_embedding = dim_embedding
        tmp = th.randn(self.dim_embedding).float()
        self.re_DTI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_DR_CH = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_DR_DR = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_PR_PR = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_PR_SEQ = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))

    def forward(self, drug_drug, drug_chemical, protein_protein, protein_sequence,
                drug_protein, drug_feature, protein_feature,
                g, neg_g):
        drug_protein_reconstruct = th.mm(th.mm(drug_feature, self.re_DTI), protein_feature.t())
        drug_drug_reconstruct = th.mm(th.mm(drug_feature, self.re_DR_DR), drug_feature.t())
        drug_ch_reconstruct = th.mm(th.mm(drug_feature, self.re_DR_CH), drug_feature.t())
        protein_protein_reconstruct = th.mm(th.mm(protein_feature, self.re_PR_PR), protein_feature.t())
        protein_seq_reconstruct = th.mm(th.mm(protein_feature, self.re_PR_SEQ), protein_feature.t())

        # DTI_potential = drug_protein_reconstruct - drug_protein.float()

        # tmp = th.mul(drug_protein_mask.float(), (drug_protein_reconstruct - drug_protein.float()))
        # drug_protein_reconstruct_loss = th.sum(tmp ** 2)
        drug_protein_reconstruct_loss = th.sum((drug_protein_reconstruct - drug_protein.float()) ** 2)
        drug_drug_reconstruct_loss = th.sum((drug_drug_reconstruct - drug_drug.float()) ** 2)
        drug_ch_reconstruct_loss = th.sum((drug_ch_reconstruct - drug_chemical.float()) ** 2)
        protein_protein_reconstruct_loss = th.sum((protein_protein_reconstruct - protein_protein.float()) ** 2)
        protein_seq_reconstruct_loss = th.sum((protein_seq_reconstruct - protein_sequence.float()) ** 2)

        other_loss = drug_drug_reconstruct_loss + drug_ch_reconstruct_loss + \
                     protein_protein_reconstruct_loss + protein_seq_reconstruct_loss
        tloss = drug_protein_reconstruct_loss + 1.0 * other_loss
        tloss = drug_protein_reconstruct_loss

        pos_src_index, pos_dst_index = g.edges()
        neg_src_index, neg_dst_index = neg_g.edges()
        pos_src_index, pos_dst_index = pos_src_index.cpu().numpy(), pos_dst_index.cpu().numpy()
        neg_src_index, neg_dst_index = neg_src_index.cpu().numpy(), neg_dst_index.cpu().numpy()
        pos_h = np.array(drug_protein_reconstruct.detach().cpu().numpy())[pos_src_index, pos_dst_index]
        neg_h = np.array(drug_protein_reconstruct.detach().cpu().numpy())[neg_src_index, neg_dst_index]
        pre, target = predict_target_pair(th.tensor(pos_h), th.tensor(neg_h))
        becloss, roc_auc, aupr = compute_score(pre, target)

        return tloss, roc_auc, aupr
