# -*- coding: utf-8 -*-
import os
import random
import time

import torch
import torch as th
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import scipy.sparse as sp
import sys

sys.path.append('../')
from model.CoGnnNet import HeCo
from model.HSGL import HSGL
from model.SHGCL import SHGCL
from layers.MLPPredicator import MLPPredicator
from tools.args import parse_args
from tools.tools import load_data, ConstructGraph, load_feature, construct_postive_graph, \
    sparse_mx_to_torch_sparse_tensor, l2_norm, concat_link, normalize_adj, compute_score, compute_loss, \
    compute_auc_aupr
from data_process.GetPos import get_pos, get_pos_identity
from tools.EarlyStopping import EarlyStopping
from tools.DTIDataSet import DTIDataSet
import warnings

warnings.filterwarnings('ignore')

# 常量
DR_DR_V = 'drug_drug virtual'
PR_PR_V = 'protein_protein virtual'
DI_DI_V = 'disease_disease virtual'
SE_SE_V = 'sideeffect_sideeffect virtual'

DR_PR_I = 'drug_protein interaction'
PR_DR_I = 'protein_drug interaction'

DR_DR_A = 'drug_drug association'
DR_PR_A = 'drug_protein association'
DR_DI_A = 'drug_disease association'
DR_SE_A = 'drug_sideeffect association'

PR_DR_A = 'protein_drug association'
PR_PR_A = 'protein_protein association'
PR_DI_A = 'protein_disease association'
PR_SE_A = 'protein_sideeffect association'

DI_DR_A = 'disease_drug association'
DI_PR_A = 'disease_protein association'
DI_DI_A = 'disease_disease association'
DI_SE_A = 'disease_sideeffect association'

SE_DR_A = 'sideeffect_drug association'
SE_PR_A = 'sideeffect_protein association'
SE_DI_A = 'sideeffect_disease association'
SE_SE_A = 'sideeffect_sideeffect association'

drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'
relation_dti = ('drug', 'drug_protein interaction', 'protein')

# 在异构转同构的过程中,num_nodes_per_ntype为[5603,708,1512,4192] 分别对应di drug pro se
drug_len = 708
protein_len = 1512
sideeffect_len = 4192
disease_len = 5603

args = parse_args()
print(args)
device = args.device

DRDRDR, DRPRDR, DRSEDR, DRDIDR = 'DRDRDR', 'DRPRDR', 'DRSEDR', 'DRDIDR'
PRDRPR, PRPRPR, PRDIPR = 'PRDRPR', 'PRPRPR', 'PRDIPR'
SEDRSE = 'SEDRSE'
DIDRDI, DIPRDI = 'DIDRDI', "DIPRDI"


def TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_drug, drug_chemical, drug_disease,
                     drug_sideeffect, protein_protein, protein_sequence, protein_disease):
    best_valid_aupr, test_aupr, test_auc, patience = 0., 0., 0., 0.
    train_label = th.tensor(DTItrain[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    val_label = th.tensor(DTIvalid[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    test_label = th.tensor(DTItest[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    num_disease = len(drug_disease.T)
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_sideeffect = len(drug_sideeffect.T)

    drug_protein = th.zeros((num_drug, num_protein))
    mask = th.zeros((num_drug, num_protein)).to(device)
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

    hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                  protein_sequence, protein_disease, drug_protein)
    dti_np = drug_protein.numpy()
    drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(np.matmul(dti_np, dti_np.T)))).to(device)
    prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(np.matmul(dti_np.T, dti_np)))).to(device)
    drdrdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drdrdr.npz'))).to(device)
    drdidr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drdidr.npz'))).to(device)
    drsedr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drsedr.npz'))).to(device)
    prprpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prprpr.npz'))).to(device)
    prdipr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prdipr.npz'))).to(device)
    didrdi = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/didrdi.npz'))).to(device)
    diprdi = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/diprdi.npz'))).to(device)
    sedrse = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/sedrse.npz'))).to(device)

    mps_dict = {drug: [drdrdr, drprdr], protein: [prdrpr, prprpr], disease: [], sideeffect: []}
    # mps_dict = {drug: [drdrdr, drprdr], protein: [prdrpr, prprpr], disease: [didrdi, diprdi], sideeffect: [sedrse]}
    pos_dict = get_pos(hetero_graph, device)
    mp_len_dict = {k: len(v) for k, v in mps_dict.items()}
    drug_dr = th.tensor(drug_drug).to(device)
    drug_ch = th.tensor(drug_chemical).to(device)
    drug_di = th.tensor(drug_disease).to(device)
    drug_se = th.tensor(drug_sideeffect).to(device)
    protein_pr = th.tensor(protein_protein).to(device)
    protein_seq = th.tensor(protein_sequence).to(device)
    protein_di = th.tensor(protein_disease).to(device)
    drug_pr = drug_protein.to(device)

    node_feature = load_feature()
    keys = [drug, protein, disease, sideeffect]
    model = SHGCL(node_feature, args.hid_dim, args, keys, mp_len_dict, args.attn_drop).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.epochs):
        model.train()
        tloss, dp_re = model(drug_dr, drug_ch, drug_di, drug_se, protein_pr,
                             protein_seq, protein_di, drug_pr, mask, mps_dict, pos_dict, args.cl,
                             node_feature)
        results = dp_re.detach()
        train_pre = results[DTItrain[:, 0], DTItrain[:, 1]]
        train_auc, train_aupr = compute_auc_aupr(train_pre, train_label)
        loss = tloss
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()
        with th.no_grad():
            val_pre = results[DTIvalid[:, 0], DTIvalid[:, 1]]
            valid_auc, valid_aupr = compute_auc_aupr(val_pre, val_label)
            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
                patience = 0
                test_pre = results[DTItest[:, 0], DTItest[:, 1]]
                test_auc, test_aupr = compute_auc_aupr(test_pre, test_label)
            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break
        if i % 25 == 0:
            print(
                "Epoch {:05d} | Train Loss {:02f} | Train auc {:.4f} | Train aupr {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f} | Test ROC_AUC {:.4f} | Test AUPR {:.4f}"
                    .format(i, loss.item(), train_auc, train_aupr, valid_auc, valid_aupr, test_auc,
                            test_aupr))

    return test_auc, test_aupr


def main():
    drug_d, drug_ch, drug_di, drug_side, protein_p, protein_seq, protein_di, dti_original = load_data()

    # sampling
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_original)[0]):
        for j in range(np.shape(dti_original)[1]):
            if int(dti_original[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_original[i][j]) == 0:
                whole_negative_index.append([i, j])

    # pos:neg=1:10
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=10 * len(whole_positive_index), replace=False)

    # All unknown DTI pairs all treated as negative examples
    '''negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(whole_negative_index), replace=False)'''

    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    test_auc_round = []
    test_aupr_round = []

    for r in range(1):
        print("----------------------------------------")

        test_auc_fold = []
        test_aupr_fold = []

        kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
        k_fold = 0

        for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
            train = data_set[train_index]
            DTItest = data_set[test_index]
            DTItrain, DTIvalid = train_test_split(train, test_size=0.05, random_state=0)

            k_fold += 1
            print("--------------------------------------------------------------")
            print("KFold ", k_fold, " of 10")
            print("--------------------------------------------------------------")

            time_roundStart = time.time()

            t_auc, t_aupr = TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_d,
                                             drug_ch, drug_di, drug_side, protein_p,
                                             protein_seq, protein_di)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)
            time_roundEnd = time.time()
            print('auc:', t_auc, 't_aupr', t_aupr)
            print("Time spent in this fold:", time_roundEnd - time_roundStart)

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        print('mean_auc:', test_auc_round, 'mean_aupr:', test_aupr_round)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Total time:", end - start)
