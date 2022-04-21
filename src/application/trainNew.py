# -*- coding: utf-8 -*-
import os

import dgl
import time
import torch as th
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import scipy.sparse as sp
from model.CoGnnNet import HeCo
from src.tools.args import parse_args, parse_argsCO
from src.tools.tools import load_data, ConstructGraph, load_feature, construct_negative_graph, construct_postive_graph, \
    sparse_mx_to_torch_sparse_tensor, l2_norm, concat_link_pos, normalize_adj
from CoDTI.src.model.GnnNetV2 import GRDTI, GnnNetV2, GnnNetV3
from CoDTI.src.model.GnnNet import GnnModel
from tools.EarlyStopping import EarlyStopping
from src.layers.MLPPredicator import MLPPredicatorDTI

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


# pre train
# pos_dict = {drug: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/drug_pos.npz')).to(device),
#             protein: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/protein_pos.npz')).to(device),
#             sideeffect: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/sideeffect_pos.npz')).to(
#                 device),
#             disease: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/disease_pos.npz')).to(device)}
# mp_dict = {drug: [[DR_DR_A, DR_DR_A], [DR_PR_I, PR_DR_I], [DR_SE_A, SE_DR_A], [DR_DI_A, DI_DR_A]],
#            protein: [[PR_DR_I, DR_PR_I], [PR_PR_A, PR_PR_A], [PR_DI_A, DI_PR_A]],
#            sideeffect: [[SE_DR_A, DR_SE_A]],
#            disease: [[DI_DR_A, DR_DI_A], [DI_PR_A, PR_DI_A]]}
# mp_key_dict = {drug: [DRDRDR, DRPRDR, DRSEDR, DRDIDR],
#                protein: [PRDRPR, PRPRPR, PRDIPR],
#                sideeffect: [SEDRSE],
#                disease: [DIDRDI, DIPRDI]}

def preTrain(DTItrain, node_feature, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
             protein_sequence, protein_disease):
    # mp_dict = {drug: [[DR_DR_A, DR_DR_A], [DR_PR_I, PR_DR_I], [DR_SE_A, SE_DR_A], [DR_DI_A, DI_DR_A]],
    #            protein: [[PR_DR_I, DR_PR_I], [PR_PR_A, PR_PR_A], [PR_DI_A, DI_PR_A]]}
    # mp_key_dict = {drug: [DRDRDR, DRPRDR, DRSEDR, DRDIDR],
    #                protein: [PRDRPR, PRPRPR, PRDIPR]}
    # mp_dict = {drug: [[DR_DR_A, DR_DR_A], [DR_PR_I, PR_DR_I], [DR_SE_A, SE_DR_A], [DR_DI_A, DI_DR_A]],
    #            protein: [[PR_DR_I, DR_PR_I], [PR_PR_A, PR_PR_A], [PR_DI_A, DI_PR_A]],
    #            sideeffect: [[SE_DR_A, DR_SE_A]],
    #            disease: [[DI_DR_A, DR_DI_A], [DI_PR_A, PR_DI_A]]}
    # mp_key_dict = {drug: [DRDRDR, DRPRDR, DRSEDR, DRDIDR],
    #                protein: [PRDRPR, PRPRPR, PRDIPR],
    #                sideeffect: [SEDRSE],
    #                disease: [DIDRDI, DIPRDI]}

    pos_dict = {drug: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/drug_pos.npz')).to(device),
                protein: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/protein_pos.npz')).to(device)}
    # pos_dict = {drug: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/drug_pos.npz')).to(device),
    #             protein: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/protein_pos.npz')).to(device),
    #             sideeffect: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/sideeffect_pos.npz')).to(
    #                 device),
    #             disease: sparse_mx_to_torch_sparse_tensor(sp.load_npz('../../data/pos/disease_pos.npz')).to(device)}
    mp_len_dict = {drug: 4, protein: 3, disease: 2, sideeffect: 1}
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

    mps_dict = {drug: [drdrdr, drprdr, drdidr, drsedr],protein: [prdrpr, prprpr, prdipr],
                disease: [didrdi, diprdi],sideeffect: [sedrse]}
    drug_protein = th.zeros((drug_len, protein_len))
    mask = th.zeros((drug_len, protein_len)).to(device)
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

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

    model.load_state_dict(th.load('checkpoint.pt'))
    os.remove('checkpoint.pt')
    model.eval()
    embeds = model.get_mp_embeds(node_feature, mps_dict)
    # embeds = model.get_sc_embeds(hetero_graph, node_feature)
    for k, v in embeds.items():
        embeds[k] = v.detach()
        embeds[k] = l2_norm(embeds[k])
    return embeds


def TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_drug, drug_chemical, drug_disease,
                     drug_sideeffect, protein_protein, protein_sequence, protein_disease, node_feature):
    in_size = len(node_feature[drug].T)
    device = th.device(args.device)
    num_disease = disease_len
    num_drug = drug_len
    num_protein = protein_len
    num_sideeffect = sideeffect_len

    drug_protein = th.zeros((num_drug, num_protein))
    mask = th.zeros((num_drug, num_protein)).to(device)
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

    best_valid_aupr = 0.
    # best_valid_auc = 0
    patience = 0.

    g = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                       protein_disease, drug_protein, args).to(device)
    negative_graph = construct_negative_graph(g, 10, relation_dti).to(device)

    drug_drug = th.tensor(drug_drug).to(device)
    drug_chemical = th.tensor(drug_chemical).to(device)
    drug_disease = th.tensor(drug_disease).to(device)
    drug_sideeffect = th.tensor(drug_sideeffect).to(device)
    protein_protein = th.tensor(protein_protein).to(device)
    protein_sequence = th.tensor(protein_sequence).to(device)
    protein_disease = th.tensor(protein_disease).to(device)
    drug_protein = drug_protein.to(device)
    edge_dict = {drug: {'drug_drug': drug_drug, 'drug_ch': drug_chemical, 'drug_disease': drug_disease,
                        'drug_sideeffect': drug_sideeffect, 'drug_protein': drug_protein},
                 protein: {'protein_protein': protein_protein, 'protein_sequence': protein_sequence,
                           'protein_disease': protein_disease, 'drug_protein': drug_protein.T},
                 sideeffect: {'drug_sideeffect': drug_sideeffect.T},
                 disease: {'drug_disease': drug_disease.T, 'protein_disease': protein_disease.T}}
    feat_dict = {drug: {'drug_drug': drug, 'drug_ch': drug, 'drug_disease': disease,
                        'drug_sideeffect': sideeffect, 'drug_protein': protein},
                 protein: {'protein_protein': protein, 'protein_sequence': protein,
                           'protein_disease': disease, 'drug_protein': drug},
                 sideeffect: {'drug_sideeffect': drug},
                 disease: {'drug_disease': drug, 'protein_disease': protein}}
    # model = GRDTI(args.hid_dim, args.out_dim)
    model = GnnNetV2(args.out_dim, args.feat_drop, edge_dict)
    # model = GnnModel(g, 128, args.out_dim, args.out_dim, args=args)
    model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    test_roc_auc = 0
    test_aupr = 0
    for i in range(args.epochs):

        model.train()
        # tloss, dp_re = model(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
        #                      protein_sequence, protein_disease, drug_protein, mask, node_feature)
        tloss, dp_re = model(node_feature, edge_dict, feat_dict, drug_drug, drug_chemical, protein_protein,
                             protein_sequence, drug_protein, drug_disease, drug_sideeffect, protein_disease, mask)
        # tloss, dp_re = model(g, negative_graph, node_feature, relation_dti,
        #                      g.edge_type_subgraph([DR_PR_I]), drug_drug,
        #                      drug_chemical, protein_protein, protein_sequence, drug_protein, mask)
        results = dp_re.detach().cpu()
        optimizer.zero_grad()
        loss = tloss
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()

        with th.no_grad():
            pred_list = []
            ground_truth = []
            for ele in DTIvalid:
                pred_list.append(results[ele[0], ele[1]])
                ground_truth.append(ele[2])
            valid_auc = roc_auc_score(ground_truth, pred_list)
            valid_aupr = average_precision_score(ground_truth, pred_list)

            pred_list_train = []
            ground_truth_train = []
            for ele in DTItrain:
                pred_list_train.append(results[ele[0], ele[1]])
                ground_truth_train.append(ele[2])
            train_auc = roc_auc_score(ground_truth_train, pred_list_train)
            train_aupr = average_precision_score(ground_truth_train, pred_list_train)

            test_pred_list = []
            test_ground_truth = []
            if valid_aupr >= best_valid_aupr:
                patience = 0
                best_valid_aupr = valid_aupr
                for ele in DTItest:
                    test_pred_list.append(results[ele[0], ele[1]])
                    test_ground_truth.append(ele[2])
                test_roc_auc = roc_auc_score(test_ground_truth, test_pred_list)
                test_aupr = average_precision_score(test_ground_truth, test_pred_list)
            else:
                patience += 1
                if patience > args.patience and i > 300:
                    print("Early Stopping")
                    break
        if i % 25 == 0:
            print(
                "Epoch {:05d} | Train Loss {:02f} | Train auc {:.4f} | Train aupr {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f} | Test ROC_AUC {:.4f} | Test AUPR {:.4f}"
                    .format(i, loss.cpu().data.numpy(), train_auc, train_aupr, valid_auc, valid_aupr, test_roc_auc,
                            test_aupr))
    return test_roc_auc, test_aupr


def TrainAndEvaluateV2(DTItrain, DTIvalid, DTItest, args, drug_drug, drug_chemical, drug_disease,
                       drug_sideeffect, protein_protein, protein_sequence, protein_disease, node_feature):
    device = th.device(args.device)
    drug_protein = th.zeros((708, 1512))
    mask = th.zeros((708, 1512)).to('cuda:3')
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

    drug_drug = th.tensor(drug_drug).to(device)
    drug_chemical = th.tensor(drug_chemical).to(device)
    drug_disease = th.tensor(drug_disease).to(device)
    drug_sideeffect = th.tensor(drug_sideeffect).to(device)
    protein_protein = th.tensor(protein_protein).to(device)
    protein_sequence = th.tensor(protein_sequence).to(device)
    protein_disease = th.tensor(protein_disease).to(device)
    drug_protein = drug_protein.to(device)
    feat_dict = {drug: {'drug_drug': drug, 'drug_ch': drug, 'drug_disease': disease,
                        'drug_sideeffect': sideeffect, 'drug_protein': protein},
                 protein: {'protein_protein': protein, 'protein_sequence': protein,
                           'protein_disease': disease, 'drug_protein': drug},
                 sideeffect: {'drug_sideeffect': drug},
                 disease: {'drug_disease': drug, 'protein_disease': protein}}
    edge_dict = {drug: {'drug_drug': drug_drug, 'drug_ch': drug_chemical, 'drug_disease': drug_disease,
                        'drug_sideeffect': drug_sideeffect, 'drug_protein': drug_protein},
                 protein: {'protein_protein': protein_protein, 'protein_sequence': protein_sequence,
                           'protein_disease': protein_disease, 'drug_protein': drug_protein.T},
                 sideeffect: {'drug_sideeffect': drug_sideeffect.T},
                 disease: {'drug_disease': drug_disease.T, 'protein_disease': protein_disease.T}}

    best_valid_aupr = 0.
    patience = 0.

    model = MLPPredicatorDTI(args.hid_dim * 2, 1)
    # model = GnnNetV3(args.out_dim, args.feat_drop, edge_dict)

    model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    test_auc = 0
    test_aupr = 0
    train_graph = construct_postive_graph((DTItrain[:, 0], DTItrain[:, 1]), relation_dti).to(device)
    val_graph = construct_postive_graph((DTIvalid[:, 0], DTIvalid[:, 1]), relation_dti).to(device)
    test_graph = construct_postive_graph((DTItest[:, 0], DTItest[:, 1]), relation_dti).to(device)

    train_h = concat_link_pos(train_graph, node_feature[drug], node_feature[protein], relation_dti)
    val_h = concat_link_pos(val_graph, node_feature[drug], node_feature[protein], relation_dti)
    test_h = concat_link_pos(test_graph, node_feature[drug], node_feature[protein], relation_dti)

    for i in range(args.epochs):

        model.train()
        loss, train_auc, train_aupr = model(DTItrain, train_h)
        # loss, train_auc, train_aupr = model(node_feature, edge_dict, feat_dict, DTItrain, train_graph)
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()

        with th.no_grad():
            val_loss, valid_auc, valid_aupr = model(DTIvalid, val_h)
            # val_loss, valid_auc, valid_aupr = model(node_feature, edge_dict, feat_dict, DTIvalid, val_graph)
            if valid_aupr >= best_valid_aupr:
                patience = 0
                best_valid_aupr = valid_aupr
                train_loss, test_auc, test_aupr = model(DTItest, test_h)
                # train_loss, test_auc, test_aupr = model(node_feature, edge_dict, feat_dict, DTItest, test_graph)
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
    for i in range(np.shape(dti_original)[0]):
        for j in range(np.shape(dti_original)[1]):
            if int(dti_original[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_original[i][j]) == 0:
                whole_negative_index.append([i, j])

    # pos:neg=1:10
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=10 * len(whole_positive_index), replace=False)
    # 测试是不是样本的问题
    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)

    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    # 测试是不是样本的问题
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

        kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        k_fold = 0

        for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
            train = data_set[train_index]
            DTItest = data_set[test_index]
            DTItrain, DTIvalid = train_test_split(train, test_size=0.1, random_state=None)

            k_fold += 1
            print("--------------------------------------------------------------")
            print("round ", r + 1, " of ", rounds, ":", "KFold ", k_fold, " of 10")
            print("--------------------------------------------------------------")
            node_feature = load_feature()
            node_feature = preTrain(DTItrain, node_feature, drug_d, drug_ch, drug_di, drug_side, protein_p, protein_seq,
                                    protein_di)
            # t_auc, t_aupr = TrainAndEvaluate(DTItrain, DTIvalid, DTItest, args, drug_d,
            #                                  drug_ch, drug_di, drug_side, protein_p,
            #                                  protein_seq, protein_di, node_feature)
            t_auc, t_aupr = TrainAndEvaluateV2(DTItrain, DTIvalid, DTItest, args, drug_d,
                                               drug_ch, drug_di, drug_side, protein_p,
                                               protein_seq, protein_di, node_feature)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        print("test_auc: ", test_auc_round, ' testaupr: ', test_aupr_round)


if __name__ == "__main__":
    main()
