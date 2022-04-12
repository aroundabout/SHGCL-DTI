import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from numpy import *
import numpy as np
import dgl

from src.model.CoGnnNet import HeCo
import os
from src.layers.MLPPredicator import MLPPredicator
from src.layers.DistMulLayer import DistLayer
from tools.tools import ConstructGraph, load_data, ConstructGraphWithRW, \
    ConstructGraphOnlyWithRW, construct_negative_graph, compute_loss, \
    construct_postive_graph, evaluate, load_feature, compute_score, \
    predict_target_pair, concat_link, sparse_mx_to_torch_sparse_tensor
from src.tools.args import parse_argsCO
from src.tools.EarlyStopping import EarlyStopping
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch as th

drug = 'drug'
protein = 'protein'
relation_dti = ('drug', 'drug_protein interaction', 'protein')
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

DRDRDR, DRPRDR, DRSEDR, DRDIDR = 'DRDRDR', 'DRPRDR', 'DRSEDR', 'DRDIDR'
PRDRPR, PRPRPR, PRDIPR = 'PRDRPR', 'PRPRPR', 'PRDIPR'
args = parse_argsCO()
print(args)
device = args.device
drug_num = 708
protein_num = 1512

mp_dict = {drug: [[DR_DR_A, DR_DR_A], [DR_PR_I, PR_DR_I], [DR_SE_A, SE_DR_A], [DR_DI_A, DI_DR_A]],
           protein: [[PR_DR_I, DR_PR_I], [PR_PR_A, PR_PR_A], [PR_DI_A, DI_PR_A]]}

mp_key_dict = {drug: [DRDRDR, DRPRDR, DRSEDR, DRDIDR],
               protein: [PRDRPR, PRPRPR, PRDIPR]}


def print_fold(index):
    print("--------------------------------------------------------------")
    print("KFold ", index, " of 10")
    print("--------------------------------------------------------------")


def val_test_eva(model, pos, neg, pos_weight):
    with torch.no_grad():
        pre, target = predict_target_pair(model(pos), model(neg))
        return compute_score(pre, target, pos_weight=pos_weight)


def evaluateDistMult(dims_emb, drug_feature, protein_feature, dti_train, dti_val, dti_test, etype, fold, index,
                     drug_drug, drug_chemical, protein_protein, protein_sequence, drug_protein, drug_protein_mask):
    model = DistLayer(dims_emb)
    model.to(device)
    dti_train = dti_train[:, 0], dti_train[:, 1]
    dti_val = dti_val[:, 0], dti_val[:, 1]
    dti_test = dti_test[:, 0], dti_test[:, 1]
    train_graph = construct_postive_graph(dti_train, etype).to(device)
    train_neg_graph = construct_negative_graph(train_graph, fold, etype).to(device)
    val_graph = construct_postive_graph(dti_val, etype).to(device)
    val_neg_graph = construct_negative_graph(val_graph, fold, etype).to(device)
    test_graph = construct_postive_graph(dti_test, etype).to(device)
    test_neg_graph = construct_negative_graph(test_graph, fold, etype).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    early_stopping = EarlyStopping(patience=args.patience)
    drug_drug, drug_chemical, protein_protein, protein_sequence, drug_protein = \
        th.tensor(drug_drug).to(device), th.tensor(drug_chemical).to(device), th.tensor(protein_protein).to(device), \
        th.tensor(protein_sequence).to(device), th.tensor(drug_protein).to(device)
    for epoch in range(args.epoch_mlp):
        model.train()
        loss, roc_auc_train, aupr_train = model(drug_drug, drug_chemical, protein_protein, protein_sequence,
                                                drug_protein, drug_protein_mask, drug_feature, protein_feature,
                                                train_graph, train_neg_graph)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
        model.eval()
        loss_val, roc_auc_val, aupr_val = model(drug_drug, drug_chemical, protein_protein, protein_sequence,
                                                drug_protein, drug_protein_mask, drug_feature, protein_feature,
                                                val_graph, val_neg_graph)
        if epoch % 25 == 0:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train ROC_AUC {:.4f} | Train AUPR {:.4f} | Val Loss {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f}"
                    .format(epoch, loss.item(), roc_auc_train, aupr_train, loss_val.item(), roc_auc_val, aupr_val))
        early_stopping(loss_val, model)
        if early_stopping.early_stop and epoch > 300:
            print("Epoch {:03d} | early stop".format(epoch))
            break
    model.eval()
    loss_test, roc_auc_test, aupr_test = model(drug_drug, drug_chemical, protein_protein, protein_sequence,
                                               drug_protein, drug_protein_mask, drug_feature, protein_feature,
                                               test_graph, test_neg_graph)

    print("--------------------------------------------------------------")
    print("Fold {:02d} | Loss {:.4f} |ROC_AUC {:.4f} | AUPR {:.4f}"
          .format(index, loss_test.item(), roc_auc_test, aupr_test))
    print("--------------------------------------------------------------")
    return loss_test, roc_auc_test, aupr_test


def evaluateMLP(embeds, out_size, dti_train, dti_val, dti_test, etype, fold, index):
    model = MLPPredicator(out_size * 2, 1)
    model = model.to(device)
    dti_train = dti_train[:, 0], dti_train[:, 1]
    dti_val = dti_val[:, 0], dti_val[:, 1]
    dti_test = dti_test[:, 0], dti_test[:, 1]
    train_graph = construct_postive_graph(dti_train, etype).to(device)
    train_neg_graph = construct_negative_graph(train_graph, fold, etype).to(device)
    val_graph = construct_postive_graph(dti_val, etype).to(device)
    val_neg_graph = construct_negative_graph(val_graph, fold, etype).to(device)
    test_graph = construct_postive_graph(dti_test, etype).to(device)
    test_neg_graph = construct_negative_graph(test_graph, fold, etype).to(device)
    pos_h, neg_h = concat_link(train_graph, train_neg_graph, embeds[drug], embeds[protein], etype)
    pos_val, neg_val = concat_link(val_graph, val_neg_graph, embeds[drug], embeds[protein], etype)
    pos_test, neg_test = concat_link(test_graph, test_neg_graph, embeds[drug], embeds[protein], etype)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    early_stopping = EarlyStopping(patience=args.patience)
    for epoch in range(args.epoch_mlp):
        model.train()
        pre, target = predict_target_pair(model(pos_h), model(neg_h))
        loss, roc_auc_train, aupr_train = compute_score(pre, target, pos_weight=torch.tensor(fold))
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
        model.eval()
        loss_val, roc_auc_val, aupr_val = val_test_eva(model, pos_val, neg_val, pos_weight=torch.tensor(fold))
        if epoch % 25 == 0:
            print(
                "Epoch {:05d} | Train Loss {:.4f} | Train ROC_AUC {:.4f} | Train AUPR {:.4f} | Val Loss {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f}"
                    .format(epoch, loss.item(), roc_auc_train, aupr_train, loss_val.item(), roc_auc_val, aupr_val))
        early_stopping(loss_val, model)
        if early_stopping.early_stop and epoch > 300:
            print("Epoch {:03d} | early stop".format(epoch))
            break
    model.eval()
    loss_test, roc_auc_test, aupr_test = val_test_eva(model, pos_test, neg_test, pos_weight=torch.tensor(fold))

    print("--------------------------------------------------------------")
    print("Fold {:02d} | Loss {:.4f} |ROC_AUC {:.4f} | AUPR {:.4f}"
          .format(index, loss_test.item(), roc_auc_test, aupr_test))
    print("--------------------------------------------------------------")
    return loss_test, roc_auc_test, aupr_test


def train():
    drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, \
    protein_sequence, protein_disease, dti_original = load_data()
    coo_data = coo_matrix(dti_original)
    data_set = np.zeros((3, len(coo_data.row)), dtype=int)
    data_set[0], data_set[1], data_set[2] = coo_data.row, coo_data.col, coo_data.data
    data_set = data_set.T

    for r in range(args.rounds):
        kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        roc_test_list = []
        aupr_test_list = []
        # 十倍交叉实验总是要先分数据集再实验
        pos_dict = {drug: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/drug_pos.npz')).to(device),
                    protein: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/protein_pos.npz')).to(device)}
        for index, (train_index, test_index) in enumerate(kf.split(data_set[:, :2], data_set[:, 2])):
            fold = 10  # 负样本倍数
            drug_protein = torch.zeros(drug_num, protein_num)
            dti_train = data_set[train_index]
            # for ele in dti_train:
            #     drug_protein[ele[0], ele[1]] = 1

            dti_train, dti_val = train_test_split(dti_train, test_size=0.05, random_state=None)
            dti_test = data_set[test_index]
            #
            for ele in dti_train:
                drug_protein[ele[0], ele[1]] = 1
            print_fold(index)
            node_features = load_feature()
            # 和端到端的实验不同 现阶段不需要创建负样本
            hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                          protein_sequence, protein_disease, drug_protein, args, CO=True).to(device)
            model = HeCo(128, args.out_dim, mp_key_dict, hetero_graph, args.attn_drop,
                         len(hetero_graph.etypes), args.tau,
                         args.lam, [drug, protein])
            mp_graph_dict = {}
            for k, lists in mp_dict.items():
                mp_graph_dict[k] = [dgl.metapath_reachable_graph(hetero_graph, mp).to(device) for mp in lists]
            model = model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
            early_stopping = EarlyStopping(patience=args.patience)

            for epoch in range(args.epochs):
                model.train()
                opt.zero_grad()
                loss = model(hetero_graph, mp_graph_dict, node_features, pos_dict, mp_key_dict, mp_dict)
                if epoch % 25 == 0:
                    print("Epoch {:03d} | Train Loss {:.4f}".format(epoch, loss.item()))
                early_stopping(loss, model)
                if early_stopping.early_stop:
                    print("Epoch {:03d} | early stop".format(epoch))
                    break
                loss.backward()
                opt.step()

            model.load_state_dict(torch.load('checkpoint.pt'))
            model.eval()
            os.remove('checkpoint.pt')

            embeds_mp = model.get_mp_embeds(mp_graph_dict, node_features, mp_key_dict, mp_dict)
            embeds_sc = model.get_sc_embeds(hetero_graph, node_features)
            l, roc_auc, aupr = evaluateMLP(embeds_mp, args.out_dim, dti_train, dti_val, dti_test, relation_dti, fold,
                                           index)
            # l, roc_auc, aupr = evaluateDistMult(128, embeds_mp[drug], embeds_mp[protein], dti_train, dti_val, dti_test,
            #                                     relation_dti, fold, index, drug_drug, drug_chemical, protein_protein,
            #                                     protein_sequence, drug_protein, drug_protein_mask)

            roc_test_list.append(roc_auc)
            aupr_test_list.append(aupr)
        mean_roc = mean(roc_test_list)
        mean_aupr = mean(aupr_test_list)
        print("--------------------------------------------------------------")
        print("--------------------------------------------------------------")
        print("Round {:02d} | ROC_AUC {:.4f} | AUPR {:.4f}"
              .format(r, mean_roc, mean_aupr))
        print("--------------------------------------------------------------")
        print("--------------------------------------------------------------")


if __name__ == "__main__":
    train()
