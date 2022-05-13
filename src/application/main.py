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
from data_process.GetPos import get_pos, get_pos_sample, get_pos_identity
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


# mlp的训练和测试函数
def train(dataloader, model, optimizer):
    model.train()
    loss_fn = 0.
    pre_all = []
    target_all = []
    for i, data in enumerate(dataloader):
        h, label = data
        pre = model(h)
        target = label
        pre_all.append(pre)
        target_all.append(target)
        loss = compute_loss(pre, target)
        loss_fn += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_all, train_auc, train_aupr = compute_score(torch.cat(pre_all, 0), torch.cat(target_all, 0))
    return loss_fn / len(dataloader), train_auc, train_aupr


def val_or_test(dataloader, model):
    model.eval()
    pre_all = []
    target_all = []
    with th.no_grad():
        for i, data in enumerate(dataloader):
            h, label = data
            pre = model(h)
            target = label
            pre_all.append(pre)
            target_all.append(target)
        pre_all = torch.cat(pre_all, 0)
        target_all = torch.cat(target_all, 0)
        loss, auc, aupr = compute_score(pre_all, target_all)
    return loss, auc, aupr


# HeCo预训练
def HeCoPreTrain(DTItrain, node_feature, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                 protein_sequence, protein_disease, fold, retrain, dir_name):
    drug_protein = th.zeros((drug_len, protein_len))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
    hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                  protein_sequence, protein_disease, drug_protein, args, CO=True).to(device)

    drdrdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drdrdr.npz'))).to(device)
    drdidr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drdidr.npz'))).to(device)
    drsedr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drsedr.npz'))).to(device)
    prprpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prprpr.npz'))).to(device)
    prdipr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prdipr.npz'))).to(device)
    didrdi = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/didrdi.npz'))).to(device)
    diprdi = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/diprdi.npz'))).to(device)
    sedrse = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/sedrse.npz'))).to(device)
    dti_np = drug_protein.numpy()
    drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(np.matmul(dti_np, dti_np.T)))).to(device)
    prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(np.matmul(dti_np.T, dti_np)))).to(device)

    mps_dict = {drug: [drdrdr, drprdr, drdidr, drsedr], protein: [prdrpr, prprpr, prdipr],
                disease: [didrdi, diprdi], sideeffect: [sedrse]}
    mp_len_dict = {drug: len(mps_dict[drug]), protein: len(mps_dict[protein])}
    pos_dict = get_pos(hetero_graph, device)

    model = HeCo(args.in_dim, args.hid_dim, mp_len_dict, hetero_graph, args.attn_drop, len(hetero_graph.etypes),
                 args.tau, args.lam, [drug, protein, sideeffect, disease], feat_drop=args.feat_drop).to(device)
    opt = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    early_stopping = EarlyStopping(patience=args.patience)
    model_dir = '../bestmodel/heco' + dir_name
    if retrain or not os.path.exists(model_dir):
        for epoch in range(args.epochs):
            model.train()
            loss = model(hetero_graph, mps_dict, node_feature, pos_dict)
            if epoch % 25 == 0:
                print("Epoch {:03d} | Train Loss {:.4f}".format(epoch, loss.item()))
            early_stopping(loss, model, model_dir)
            if early_stopping.early_stop:
                print("Epoch {:03d} | early stop".format(epoch))
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.load_state_dict(th.load(model_dir))
    model.eval()
    # 从mp视角获取最后的特征输出
    with torch.no_grad():
        embeds = model.get_mp_embeds(node_feature, mps_dict)
        for k, v in embeds.items():
            embeds[k] = v.detach()
            embeds[k] = l2_norm(embeds[k])
    return embeds


# HeCo预训练
def HeCoPreTrain1(DTItrain, node_feature, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                  protein_sequence, protein_disease, fold, retrain, dir_name):
    drug_protein = th.zeros((drug_len, protein_len))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
    hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                  protein_sequence, protein_disease, drug_protein, args, CO=True).to(device)
    drug_pos, protein_pos = get_pos(hetero_graph)
    # drug_pos, protein_pos = get_pos_identity()
    # temp_meta_path = [[DR_DR_A, DR_DR_A], [DR_PR_I, PR_DR_I],
    #                   [DR_SE_A, SE_DR_A], [DR_DI_A, DI_DR_A]]
    # drug_pos_new = get_pos_sample(hetero_graph, temp_meta_path, drug_len, 20)

    dti_np = drug_protein.numpy()
    drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(np.matmul(dti_np, dti_np.T)))).to(device)
    # drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drprdr.npz'))).to(device)
    drdrdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drdrdr.npz'))).to(device)
    drdidr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drdidr.npz'))).to(device)
    drsedr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drsedr.npz'))).to(device)
    mps_dict = {drug: [drdrdr, drprdr, drdidr, drsedr]}
    # mps_dict = {drug: [drdrdr, drprdr]}
    mp_len_dict = {drug: len(mps_dict[drug])}
    pos_dict = {drug: sparse_mx_to_torch_sparse_tensor(drug_pos).to(device)}

    model = HeCo(args.in_dim, args.hid_dim, mp_len_dict, hetero_graph, args.attn_drop, len(hetero_graph.etypes),
                 args.tau, args.lam, [drug, protein, sideeffect, disease], feat_drop=args.feat_drop).to(device)
    opt = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    early_stopping = EarlyStopping(patience=args.patience)
    model_dir = '../bestmodel/heco' + dir_name
    hetero_graph = hetero_graph.edge_type_subgraph([DR_DR_A, DR_PR_I, DR_SE_A, DR_DI_A])
    if retrain or not os.path.exists(model_dir):
        for epoch in range(args.epochs):
            model.train()
            loss = model(hetero_graph, mps_dict, node_feature, pos_dict)
            if epoch % 25 == 0:
                print("Epoch {:03d} | Train Loss {:.4f}".format(epoch, loss.item()))
            early_stopping(loss, model, model_dir)
            if early_stopping.early_stop:
                print("Epoch {:03d} | early stop".format(epoch))
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.load_state_dict(th.load(model_dir))
    model.eval()
    # 从mp视角获取最后的特征输出
    with torch.no_grad():
        embeds = model.get_mp_embeds(node_feature, mps_dict)
        for k, v in embeds.items():
            embeds[k] = v.detach()
            embeds[k] = l2_norm(embeds[k])
    return embeds


def HeCoPreTrain2(DTItrain, node_feature, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                  protein_sequence, protein_disease, fold, retrain, dir_name):
    drug_protein = th.zeros((drug_len, protein_len))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
    hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                  protein_sequence, protein_disease, drug_protein, args, CO=True).to(device)
    drug_pos, protein_pos = get_pos(hetero_graph)
    # drug_pos, protein_pos = get_pos_identity()
    dti_np = drug_protein.numpy()
    prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(np.matmul(dti_np.T, dti_np)))).to(device)

    # prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prdrpr.npz'))).to(device)
    prprpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prprpr.npz'))).to(device)
    prdipr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prdipr.npz'))).to(device)

    mps_dict = {protein: [prdrpr, prprpr, prdipr]}
    # mps_dict = {protein: [prdrpr, prprpr]}

    mp_len_dict = {protein: len(mps_dict[protein])}
    pos_dict = {protein: sparse_mx_to_torch_sparse_tensor(protein_pos).to(device)}

    model = HeCo(args.in_dim, args.hid_dim, mp_len_dict, hetero_graph, args.attn_drop, len(hetero_graph.etypes),
                 args.tau, args.lam, [drug, protein, sideeffect, disease], feat_drop=args.feat_drop).to(device)
    opt = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    early_stopping = EarlyStopping(patience=args.patience)
    model_dir = '../bestmodel/heco' + dir_name
    hetero_graph = hetero_graph.edge_type_subgraph([PR_DR_I, PR_PR_A, PR_DI_A])
    if retrain or not os.path.exists(model_dir):
        for epoch in range(args.epochs):
            model.train()
            loss = model(hetero_graph, mps_dict, node_feature, pos_dict)
            if epoch % 25 == 0:
                print("Epoch {:03d} | Train Loss {:.4f}".format(epoch, loss.item()))
            early_stopping(loss, model, model_dir)
            if early_stopping.early_stop:
                print("Epoch {:03d} | early stop".format(epoch))
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.load_state_dict(th.load(model_dir))
    model.eval()
    # 从mp视角获取最后的特征输出
    with torch.no_grad():
        embeds = model.get_mp_embeds(node_feature, mps_dict)
        for k, v in embeds.items():
            embeds[k] = v.detach()
            embeds[k] = l2_norm(embeds[k])
    return embeds


def MLPLinkPred(DTItrain, DTIvalid, DTItest, args, node_feature, fold, retrain, dir_name):
    best_valid_aupr, patience, test_auc, test_aupr = 0., 0., 0., 0.
    train_h = concat_link(DTItrain, node_feature[drug], node_feature[protein])
    val_h = concat_link(DTIvalid, node_feature[drug], node_feature[protein])
    test_h = concat_link(DTItest, node_feature[drug], node_feature[protein])
    train_label = th.tensor(DTItrain[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    val_label = th.tensor(DTIvalid[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    test_label = th.tensor(DTItest[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    train_set, val_set, test_set = \
        DTIDataSet(train_h, train_label), DTIDataSet(val_h, val_label), DTIDataSet(test_h, test_label)
    train_loader = DataLoader(dataset=train_set, batch_size=2048, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_set, batch_size=2048, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_set, batch_size=2048, shuffle=True, num_workers=0)

    model = MLPPredicator(args.hid_dim * 2, 1).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    model_dir = '../bestmodel/mlp' + dir_name
    if retrain or not os.path.exists(model_dir):
        for epoch in range(args.epochs):
            loss, train_auc, train_aupr = train(dataloader=train_loader, model=model, optimizer=optimizer)
            valid_loss, valid_auc, valid_aupr = val_or_test(dataloader=val_loader, model=model)
            if valid_aupr >= best_valid_aupr:
                patience = 0
                best_valid_aupr = valid_aupr
                test_loss, test_auc, test_aupr = val_or_test(dataloader=test_loader, model=model)
                torch.save(model.state_dict(), model_dir)
            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break
            if epoch % 25 == 0:
                print(
                    "Epoch {:05d} | Train Loss {:02f} | Train auc {:.4f} | Train aupr {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f} | Test ROC_AUC {:.4f} | Test AUPR {:.4f}"
                        .format(epoch, loss, train_auc, train_aupr, valid_auc, valid_aupr, test_auc, test_aupr))
    model.load_state_dict(th.load(model_dir))
    model.eval()
    with torch.no_grad():
        pre_test = model(test_h)
        target_test = th.tensor(DTItest[:, 2], dtype=th.float).reshape(-1, 1).to(device)
        test_loss, test_auc, test_aupr = compute_score(pre_test, target_test)
        print("Test ROC_AUC {:.4f} | Test AUPR {:.4f}".format(test_auc, test_aupr))
    return test_auc, test_aupr


def train_hgsl(model, opt, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
               protein_sequence, protein_disease, drug_protein, drug_protein_mask, node_feature: dict,
               DTItrain, train_label):
    model.train()
    loss, dti_re = model(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                         protein_sequence, protein_disease, drug_protein, drug_protein_mask)
    dti_re = dti_re.detach()
    pre = dti_re[DTItrain[:, 0], DTItrain[:, 1]]
    train_auc, train_aupr = compute_auc_aupr(pre, train_label)
    opt.zero_grad()
    loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()

    return loss.item(), train_auc, train_aupr, dti_re


def test_val_hgsl(dti_re, DTI, label):
    with torch.no_grad():
        pre = dti_re[DTI[:, 0], DTI[:, 1]]
        test_auc, test_aupr = compute_auc_aupr(pre, label)
        return test_auc, test_aupr


def hgslPred(DTItrain, DTIval, DTItest, node_feature, drug_drug, drug_chemical, drug_disease, drug_sideeffect,
             protein_protein, protein_sequence, protein_disease, retrain, dir_name):
    model_dir = '../bestmodel/hsgl' + dir_name
    best_valid_aupr, patience, test_auc, test_aupr = 0., 0., 0., 0.
    drug_protein = th.zeros((drug_len, protein_len))
    mask = th.zeros((drug_len, protein_len)).to(device)
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                  protein_sequence, protein_disease, drug_protein, args, CO=True).to(device)
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
    drug_dr = th.tensor(drug_drug).to(device)
    drug_ch = th.tensor(drug_chemical).to(device)
    drug_di = th.tensor(drug_disease).to(device)
    drug_se = th.tensor(drug_sideeffect).to(device)
    protein_pr = th.tensor(protein_protein).to(device)
    protein_seq = th.tensor(protein_sequence).to(device)
    protein_di = th.tensor(protein_disease).to(device)
    drug_pr = drug_protein.to(device)
    drug_pos, protein_pos = get_pos(hetero_graph)
    mps_dict = {drug: [drdrdr, drprdr, drdidr, drsedr], protein: [prdrpr, prprpr, prdipr]}
    mp_len_dict = {drug: len(mps_dict[drug]), protein: len(mps_dict[protein])}
    pos_dict = {drug: sparse_mx_to_torch_sparse_tensor(drug_pos).to(device),
                protein: sparse_mx_to_torch_sparse_tensor(protein_pos).to(device)}
    train_label = th.tensor(DTItrain[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    val_label = th.tensor(DTIval[:, 2], dtype=th.float).reshape(-1, 1).to(device)
    test_label = th.tensor(DTItest[:, 2], dtype=th.float).reshape(-1, 1).to(device)

    model = SHGCL(node_feature, args.hid_dim, args).to(device)
    opt = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if retrain or not os.path.exists(model_dir):
        for epoch in range(args.epochs):
            train_loss, train_auc, train_aupr, dti_re = \
                train_hgsl(model, opt, drug_dr, drug_ch, drug_di, drug_se, protein_pr,
                           protein_seq, protein_di, drug_pr, mask, node_feature, DTItrain,
                           train_label)
            valid_auc, valid_aupr = test_val_hgsl(dti_re, DTIval, val_label)
            if valid_aupr > best_valid_aupr:
                best_valid_aupr = valid_aupr
                patience = 0
                test_auc, test_aupr = test_val_hgsl(dti_re, DTItest, test_label)
                torch.save(model.state_dict(), model_dir)
            else:
                patience += 1
                if patience > args.patience:
                    print("early stop")
                    break
            if epoch % 25 == 0:
                print(
                    "Epoch {:05d} | Train Loss {:02f} | Train auc {:.4f} | Train aupr {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f} | Test ROC_AUC {:.4f} | Test AUPR {:.4f}"
                        .format(epoch, train_loss, train_auc, train_aupr, valid_auc, valid_aupr, test_auc,
                                test_aupr))

    model.load_state_dict(th.load(model_dir))
    model.eval()
    with torch.no_grad():
        loss, dti_re = model(drug_dr, drug_ch, drug_di, drug_se, protein_pr,
                             protein_seq, protein_di, drug_pr, mask)
        test_auc, test_aupr = test_val_hgsl(dti_re, DTItest, test_label)
        print("Test ROC_AUC {:.4f} | Test AUPR {:.4f}".format(test_auc, test_aupr))
    return test_auc, test_aupr


def main(random_seed, task_name, dti_path='mat_drug_protein.txt', retrain=True):
    drug_d, drug_ch, drug_di, drug_side, protein_p, protein_seq, protein_di, dti_original = \
        load_data(dti_path=dti_path)
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_original)[0]):
        for j in range(np.shape(dti_original)[1]):
            if int(dti_original[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_original[i][j]) == 0:
                whole_negative_index.append([i, j])
    if args.number == 'ten':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=10 * len(whole_positive_index), replace=False)
        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    else:
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=len(whole_negative_index), replace=False)
        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)

    count = 0
    for i in whole_positive_index:
        data_set[count][0], data_set[count][1], data_set[count][2] = i[0], i[1], 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0], data_set[count][1], data_set[count][2] = \
            whole_negative_index[i][0], whole_negative_index[i][1], 0
        count += 1

    print("----------------------------------------")
    print('random_seed=', str(random_seed), 'task=', task_name)
    print("----------------------------------------")
    test_auc_fold = []
    test_aupr_fold = []
    kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for index, (train_index, test_index) in enumerate(kf.split(data_set[:, :2], data_set[:, 2])):
        print("----------------------------------------")
        print('fold=', str(index))
        print("----------------------------------------")
        train_all, DTItest = data_set[train_index], data_set[test_index]
        DTItrain, DTIvalid = train_test_split(train_all, random_state=0, test_size=0.05)
        dir_name = '_task_' + task_name + '_rs' + str(random_seed) + '_fold' + str(index) + '.pt'
        node_feature = load_feature()
        # node_feature1 = HeCoPreTrain1(DTItrain, node_feature, drug_d, drug_ch, drug_di, drug_side, protein_p,
        #                               protein_seq, protein_di, index, retrain, dir_name)
        # node_feature2 = HeCoPreTrain2(DTItrain, node_feature, drug_d, drug_ch, drug_di, drug_side, protein_p,
        #                               protein_seq, protein_di, index, retrain, dir_name)
        # node_feature = {drug: node_feature1[drug], protein: node_feature2[protein]}
        # t_auc, t_aupr = MLPLinkPred(DTItrain, DTIvalid, DTItest, args, node_feature, index, retrain, dir_name)
        t_auc, t_aupr = hgslPred(DTItrain, DTIvalid, DTItest, node_feature, drug_d, drug_ch, drug_di, drug_side,
                                 protein_p, protein_seq, protein_di, retrain, dir_name)
        test_auc_fold.append(t_auc)
        test_aupr_fold.append(t_aupr)
    test_auc_mean = np.mean(test_auc_fold)
    test_aupr_mean = np.mean(test_aupr_fold)
    print("test_auc: ", test_auc_mean, ' testaupr: ', test_aupr_mean)
    return test_auc_mean, test_aupr_mean


def main_unique(random_seed, task_name, dti_path='mat_drug_protein.txt', retrain=True):
    drug_d, drug_ch, drug_di, drug_side, protein_p, protein_seq, protein_di, dti_original = \
        load_data(dti_path='mat_drug_protein_unique.txt')
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
    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0], data_set[count][1], data_set[count][2] = i[0], i[1], 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0], data_set[count][1], data_set[count][2] = \
            whole_negative_index[i][0], whole_negative_index[i][1], 0
        count += 1

    whole_positive_index_test = []
    whole_negative_index_test = []
    for i in range(np.shape(dti_original)[0]):
        for j in range(np.shape(dti_original)[1]):
            if int(dti_original[i][j]) == 3:
                whole_positive_index_test.append([i, j])
            elif int(dti_original[i][j] == 2):
                whole_negative_index_test.append([i, j])
    negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),
                                                  size=10 * len(whole_positive_index_test), replace=False)
    data_set_test = np.zeros((len(negative_sample_index_test) + len(whole_positive_index_test), 3), dtype=int)
    count = 0
    for i in whole_positive_index_test:
        data_set_test[count][0], data_set_test[count][1], data_set_test[count][2] = i[0], i[1], 1
        count += 1
    for i in negative_sample_index_test:
        data_set_test[count][0], data_set_test[count][1], data_set_test[count][2] = \
            whole_negative_index_test[i][0], whole_negative_index_test[i][1], 0
        count += 1

    print("----------------------------------------")
    print('random_seed=', str(random_seed), 'task=', task_name)
    print("----------------------------------------")
    test_auc_fold = []
    test_aupr_fold = []
    print("----------------------------------------")
    print('fold=', str(0))
    print("----------------------------------------")
    train_all, DTItest = data_set, data_set_test
    DTItrain, DTIvalid = train_test_split(train_all, test_size=0.05, random_state=0)
    dir_name = '_task_' + task_name + '_rs' + str(random_seed) + '_fold' + str(0) + '.pt'
    node_feature = load_feature()
    # node_feature = HeCoPreTrain(DTItrain, node_feature, drug_d, drug_ch, drug_di, drug_side, protein_p,
    #                             protein_seq, protein_di, 0, retrain, dir_name)
    t_auc, t_aupr = MLPLinkPred(DTItrain, DTIvalid, DTItest, args, node_feature, 0, retrain, dir_name)
    test_auc_fold.append(t_auc)
    test_aupr_fold.append(t_aupr)
    test_auc_mean = np.mean(test_auc_fold)
    test_aupr_mean = np.mean(test_aupr_fold)
    print("test_auc: ", test_auc_mean, ' testaupr: ', test_aupr_mean)
    return test_auc_mean, test_aupr_mean


def setup_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seeds = [88, 107, 15, 16, 21, 22, 35, 36, 47, 48, 55, 56]
    task = 'testnew'
    print("----------------------------------------")
    print('task=', task)
    print("----------------------------------------")
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # file_name = ('' if task == 'benchmark' else '_' + task)
    file_name = ''
    print(now_time)
    with open('../../result/' + task + '_auc', 'a') as f:
        f.write(now_time + '\n')
    with open('../../result/' + task + '_aupr', 'a') as f:
        f.write(now_time + '\n')
    for s in seeds:
        start = time.time()
        setup_seed(s)
        if task == 'unique':
            auc, aupr = main_unique(s, task, dti_path='mat_drug_protein' + file_name + '.txt')
        else:
            auc, aupr = main(s, task, dti_path='mat_drug_protein' + file_name + '.txt')
        with open('../../result/' + task + '_auc', 'a') as f:
            f.write(str(auc) + '\n')
        with open('../../result/' + task + '_aupr', 'a') as f:
            f.write(str(aupr) + '\n')
        end = time.time()
        print("Total time:", end - start)
