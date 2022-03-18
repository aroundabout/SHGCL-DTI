import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.data.utils import generate_mask_tensor
import numpy
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.model.GnnNet import GnnModel
from dgl.nn.pytorch.conv.gatconv import GATConv
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
from src.layers.SimpleHGNNew import SimpleHGNNew
from tools.tools import ConstructGraph, load_data, ConstructGraphWithRW, \
    ConstructGraphOnlyWithRW, construct_negative_graph, compute_loss, predict_target_pair, \
    construct_test_graph, evaluate

DR_PR_I = 'drug_protein interaction'
PR_DR_I = 'protein_drug interaction'

drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, protein_disease, dti_original = load_data()

whole_positive_index = []
for i in range(np.shape(dti_original)[0]):
    for j in range(np.shape(dti_original)[1]):
        if int(dti_original[i][j]) == 1:
            whole_positive_index.append([i, j])

data_set = np.zeros((len(whole_positive_index), 3), dtype=int)
count = 0
for i, j in whole_positive_index:
    data_set[count][0] = i
    data_set[count][1] = j
    data_set[count][2] = 1
    count += 1

kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
k_fold = 0
for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
    fold = 10
    dti_train = data_set[train_index]
    DTItrain, DTIvalid = train_test_split(dti_train, test_size=0.05, random_state=None)
    dti_val = DTIvalid[:, 0], DTIvalid[:, 1]
    dti_test = data_set[:, 0], data_set[:, 1]
    drug_protein = torch.zeros((708, 1512))
    for ele in dti_train:
        drug_protein[ele[0], ele[1]] = ele[2]
    print("--------------------------------------------------------------")
    print("KFold ", k_fold, " of 10")
    print("--------------------------------------------------------------")
    # 构建异质图
    hetero_graph = ConstructGraphWithRW(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                        protein_sequence,
                                        protein_disease, drug_protein)

    drug_feats = hetero_graph.nodes['drug'].data['feature']
    protein_feats = hetero_graph.nodes['protein'].data['feature']
    disease_feats = hetero_graph.nodes['disease'].data['feature']
    sideeffect_feats = hetero_graph.nodes['sideeffect'].data['feature']
    node_features = {'drug': drug_feats, 'protein': protein_feats, 'disease': disease_feats,
                     'sideeffect': sideeffect_feats}

    negative_graph = construct_negative_graph(hetero_graph, fold, ('drug', 'drug_protein interaction', 'protein'),
                                              node_features)

    test_graph = construct_test_graph(dti_test, ('drug', DR_PR_I, 'protein'), node_features)
    test_neg_graph = construct_negative_graph(test_graph, fold, ('drug', 'drug_protein interaction', 'protein'),
                                              node_features)

    val_graph = construct_test_graph(dti_val, ('drug', DR_PR_I, 'protein'), node_features)
    val_neg_graph = construct_negative_graph(val_graph, fold, ('drug', 'drug_protein interaction', 'protein'),
                                             node_features)

    model = GnnModel(hetero_graph, 128, 256, 128, args=None)
    opt = torch.optim.Adam(model.parameters())
    print("train start")

    for epoch in range(100):
        print("now is the ", epoch, " epoch")

        pos_score, neg_score, node_emb = \
            model(hetero_graph, negative_graph, node_features, ('drug', 'drug_protein interaction', 'protein'))

        pre, target = predict_target_pair(pos_score, neg_score)
        loss = compute_loss(pre, target)

        roc_auc_test, f1_test = \
            evaluate(model, test_graph, test_neg_graph, node_features, ('drug', 'drug_protein interaction', 'protein'))

        roc_auc_val, f1_val = \
            evaluate(model, val_graph, val_neg_graph, node_features, ('drug', 'drug_protein interaction', 'protein'))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('loss:', loss.item(), ' roc_auc_val: ', roc_auc_val, ' f1_score_val: ', f1_val,
              ' roc_auc_test: ', roc_auc_test, ' f1_test: ', f1_test)

    node_embeddings = model.sage(hetero_graph, node_features)
    print(node_embeddings)
