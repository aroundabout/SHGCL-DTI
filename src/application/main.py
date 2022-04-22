# -*- coding: utf-8 -*-
import os

import torch as th
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
import scipy.sparse as sp

from src.model.CoGnnNet import HeCo
from src.layers.MLPPredicator import MLPPredicatorDTI
from src.tools.args import parse_argsCO
from src.tools.tools import load_data, ConstructGraph, load_feature, construct_postive_graph, \
    sparse_mx_to_torch_sparse_tensor, l2_norm, concat_link_pos, normalize_adj
from src.tools.EarlyStopping import EarlyStopping

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

args = parse_argsCO()
print(args)
device = args.device

DRDRDR, DRPRDR, DRSEDR, DRDIDR = 'DRDRDR', 'DRPRDR', 'DRSEDR', 'DRDIDR'
PRDRPR, PRPRPR, PRDIPR = 'PRDRPR', 'PRPRPR', 'PRDIPR'
SEDRSE = 'SEDRSE'
DIDRDI, DIPRDI = 'DIDRDI', "DIPRDI"

# HeCo预训练
def HeCoPreTrain(DTItrain, node_feature, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                 protein_sequence, protein_disease):
    # load对比学习过程中的正样本 分为drug和protein
    pos_dict = {drug: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/drug_pos.npz')).to(device),
                protein: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/protein_pos.npz')).to(device)}
    mp_len_dict = {drug: 4, protein: 3, disease: 2, sideeffect: 1}
    # load 元路径邻接矩阵 但是邻接节点不是1 而是根据HeCo原文的方法处理了一下 处理方法在normalize_adj上
    drdrdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drdrdr.npz'))).to(device)
    drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drprdr.npz'))).to(device)
    drdidr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drdidr.npz'))).to(device)
    drsedr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/drsedr.npz'))).to(device)
    prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prdrpr.npz'))).to(device)
    prprpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prprpr.npz'))).to(device)
    prdipr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/prdipr.npz'))).to(device)
    didrdi = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/didrdi.npz'))).to(device)
    diprdi = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/diprdi.npz'))).to(device)
    sedrse = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.load_npz('../../data/mp/sedrse.npz'))).to(device)

    # 输入model的元路径dict
    mps_dict = {drug: [drdrdr, drprdr, drdidr, drsedr], protein: [prdrpr, prprpr, prdipr],
                disease: [didrdi, diprdi], sideeffect: [sedrse]}
    # 构建图中的dti从DTItrain中得到
    drug_protein = th.zeros((drug_len, protein_len))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]

    hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                  protein_sequence, protein_disease, drug_protein, args, CO=True).to(device)

    model = HeCo(args.in_dim, args.hid_dim, mp_len_dict, hetero_graph, args.attn_drop, len(hetero_graph.etypes),
                 args.tau, args.lam, [drug, protein, sideeffect, disease], feat_drop=args.feat_drop).to(device)

    opt = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    early_stopping = EarlyStopping(patience=args.patience)
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = model(hetero_graph, mps_dict, node_feature, pos_dict)
        if epoch % 25 == 0:
            print("Epoch {:03d} | Train Loss {:.4f}".format(epoch, loss.item()))
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Epoch {:03d} | early stop".format(epoch))
            break
        loss.backward()
        opt.step()
    # 在Earlystop过程中存的checkpoint.pt
    model.load_state_dict(th.load('checkpoint.pt'))
    os.remove('checkpoint.pt')
    model.eval()
    # 从mp视角获取最后的特征输出
    embeds = model.get_mp_embeds(node_feature, mps_dict)
    for k, v in embeds.items():
        embeds[k] = v.detach()
        embeds[k] = l2_norm(embeds[k])
    return embeds


def MLPLinkPred(DTItrain, DTIvalid, DTItest, args, node_feature):
    best_valid_aupr = 0.
    patience = 0.

    model = MLPPredicatorDTI(args.hid_dim * 2, 1)

    model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    test_auc = 0
    test_aupr = 0
    # 使用DTITrain,test,val去构建dgl上的图方便后面的特征拼接 这里DTI既有pos和neg样本
    train_graph = construct_postive_graph((DTItrain[:, 0], DTItrain[:, 1]), relation_dti).to(device)
    val_graph = construct_postive_graph((DTIvalid[:, 0], DTIvalid[:, 1]), relation_dti).to(device)
    test_graph = construct_postive_graph((DTItest[:, 0], DTItest[:, 1]), relation_dti).to(device)
    # drug protein拼接
    train_h = concat_link_pos(train_graph, node_feature[drug], node_feature[protein], relation_dti)
    val_h = concat_link_pos(val_graph, node_feature[drug], node_feature[protein], relation_dti)
    test_h = concat_link_pos(test_graph, node_feature[drug], node_feature[protein], relation_dti)

    for i in range(args.epochs):

        model.train()
        loss, train_auc, train_aupr = model(DTItrain, train_h)
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()

        with th.no_grad():
            # earlystop是根据aupr作为指标 在valaupr最好的时候去计算测试集
            val_loss, valid_auc, valid_aupr = model(DTIvalid, val_h)
            if valid_aupr >= best_valid_aupr:
                patience = 0
                best_valid_aupr = valid_aupr
                test_loss, test_auc, test_aupr = model(DTItest, test_h)
            else:
                patience += 1
                if patience > args.patience and i > 300:
                    print("Early Stopping")
                    break
        if i % 25 == 0:
            print(
                "Epoch {:05d} | Train Loss {:02f} | Train auc {:.4f} | Train aupr {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f} | Test ROC_AUC {:.4f} | Test AUPR {:.4f}"
                    .format(i, loss.cpu().data.numpy(), train_auc, train_aupr, valid_auc, valid_aupr, test_auc,
                            test_aupr))
    return test_auc, test_aupr


def main():
    drug_d, drug_ch, drug_di, drug_side, protein_p, protein_seq, protein_di, dti_original = load_data()
    # sampling
    whole_positive_index = []
    whole_negative_index = []
    # 从原始的dti获取正负样本
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

    rounds = args.rounds
    for r in range(rounds):
        print("----------------------------------------")

        test_auc_fold = []
        test_aupr_fold = []
        # 十倍交叉验证
        kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        k_fold = 0

        for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
            train = data_set[train_index]
            DTItest = data_set[test_index]
            # 在划分之后的train上再划分训练和验证集
            DTItrain, DTIvalid = train_test_split(train, test_size=0.05, random_state=None)

            k_fold += 1
            print("--------------------------------------------------------------")
            print("round ", r + 1, " of ", rounds, ":", "KFold ", k_fold, " of 10")
            print("--------------------------------------------------------------")
            node_feature = load_feature()
            # HeCo 预训练
            node_feature = HeCoPreTrain(DTItrain, node_feature, drug_d, drug_ch, drug_di, drug_side, protein_p, protein_seq,
                                        protein_di)
            # MLP练级预测
            t_auc, t_aupr = MLPLinkPred(DTItrain, DTIvalid, DTItest, args, node_feature)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        print("test_auc: ", test_auc_round, ' testaupr: ', test_aupr_round)


if __name__ == "__main__":
    main()
