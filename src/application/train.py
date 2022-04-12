import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from numpy import *
import numpy as np
import dgl
import torch as th
from src.model.GnnNet import GnnModel
from tools.Sampler import NegativeSampler
from tools.tools import ConstructGraph, load_data, ConstructGraphWithRW, \
    ConstructGraphOnlyWithRW, construct_negative_graph, compute_loss, \
    construct_postive_graph, evaluate, load_feature, compute_score, predict_target_pair
from src.tools.args import parse_args
from src.tools.EarlyStopping import EarlyStopping

drug = 'drug'
protein = 'protein'
DR_PR_I = 'drug_protein interaction'
relation_dti = ('drug', 'drug_protein interaction', 'protein')


def train(arg):
    device = arg.device

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

    for r in range(arg.rounds):
        kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
        roc_test_list = []
        f1_test_list = []
        aupr_test_list = []
        for index, (train_index, test_index) in enumerate(kf.split(data_set[:, :2], data_set[:, 2])):
            fold = 10
            pos_weight = torch.tensor(fold)
            dti_train = data_set[train_index]
            DTItrain, DTIvalid = train_test_split(dti_train, test_size=0.10, random_state=None)
            dti_val = DTIvalid[:, 0], DTIvalid[:, 1]
            dti_test = data_set[test_index]
            dti_test = dti_test[:, 0], dti_test[:, 1]
            drug_protein = torch.zeros((708, 1512))
            for ele in DTItrain:
                drug_protein[ele[0], ele[1]] = ele[2]
            print("--------------------------------------------------------------")
            print("KFold ", index, " of 10")
            print("--------------------------------------------------------------")
            # 构建异质图
            node_features = load_feature()
            if arg.random_walk:
                hetero_graph = ConstructGraphWithRW(drug_drug, drug_chemical, drug_disease, drug_sideeffect,
                                                    protein_protein,
                                                    protein_sequence, protein_disease, drug_protein,
                                                    arg).to(device)
            else:
                hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                              protein_sequence, protein_disease, drug_protein,
                                              arg).to(device)
            negative_graph = construct_negative_graph(hetero_graph, fold, relation_dti).to(device)

            test_graph = construct_postive_graph(dti_test, relation_dti).to(device)
            test_neg_graph = construct_negative_graph(test_graph, fold, relation_dti).to(device)

            val_graph = construct_postive_graph(dti_val, relation_dti).to(device)
            val_neg_graph = construct_negative_graph(val_graph, fold, relation_dti).to(device)

            model = GnnModel(hetero_graph, 128, arg.out_dim, arg.out_dim, args=arg)
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2)
            early_stopping = EarlyStopping(patience=arg.patience)

            dgl.dataloading.negative_sampler.Uniform(10)
            # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
            # dataloader = dgl.dataloading.EdgeDataLoader(
            #     hetero_graph, {DR_PR_I: hetero_graph.edges[DR_PR_I]}, sampler,
            #     negative_sampler=dgl.dataloading.negative_sampler.Uniform(10),
            #     batch_size=1024,
            #     shuffle=True,
            #     drop_last=False)
            h = {}
            # drug_drug, drug_chemical, protein_protein, protein_sequence, drug_protein = \
            #     th.tensor(drug_drug).to(device), th.tensor(drug_chemical).to(device), th.tensor(protein_protein).to(
            #         device), th.tensor(protein_sequence).to(device), th.tensor(drug_protein).to(device)
            for epoch in range(arg.epochs):
                model.train()

                pre, target, h = model(hetero_graph, negative_graph, node_features, relation_dti)
                loss, roc_auc_train, aupr_train = compute_score(pre, target, pos_weight=pos_weight)

                # loss, roc_auc_train, aupr_train = model(hetero_graph, negative_graph, node_features, relation_dti,
                #                                         hetero_graph.edge_type_subgraph([DR_PR_I]), drug_drug,
                #                                         drug_chemical, protein_protein, protein_sequence, drug_protein)
                model.eval()

                roc_auc_val, aupr_val, f1_val, loss_val = evaluate(model, val_graph, val_neg_graph, h[drug], h[protein],
                                                                   relation_dti, pos_weight=pos_weight)
                # loss_val, roc_auc_val, aupr_val = model(hetero_graph, val_neg_graph, node_features,
                #                                         relation_dti, val_graph, drug_drug,
                #                                         drug_chemical, protein_protein, protein_sequence, drug_protein)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if epoch % 10 == 0:
                    print(
                        "Epoch {:05d} | Train Loss {:.4f} | Train ROC_AUC {:.4f} | Train AUPR {:.4f} | Val Loss {:.4f} | Val ROC_AUC {:.4f} | Val AUPR {:.4f}"
                            .format(epoch, loss.item(), roc_auc_train, aupr_train, loss_val.item(), roc_auc_val,
                                    aupr_val))
                early_stopping(loss_val, model)
                if early_stopping.early_stop and epoch > 200:
                    print("Epoch {:02d} | early stop".format(epoch))
                    break
            print("--------------------------------------------------------------")
            model.eval()

            roc_auc_test, aupr_test, f1_test, loss_test = evaluate(model, test_graph, test_neg_graph, h[drug],
                                                                   h[protein], relation_dti, pos_weight=pos_weight)
            # loss_test, roc_auc_test, aupr_test = model(hetero_graph, test_neg_graph, node_features,
            #                                            relation_dti, test_graph, drug_drug,
            #                                            drug_chemical, protein_protein, protein_sequence, drug_protein)

            print("Fold {:02d} | Loss {:.4f} |ROC_AUC {:.4f} | AUPR {:.4f}"
                  .format(index, loss_test.item(), roc_auc_test, aupr_test))
            print("--------------------------------------------------------------")
            roc_test_list.append(roc_auc_test)
            aupr_test_list.append(aupr_test)
        mean_roc = mean(roc_test_list)
        mean_f1 = mean(f1_test_list)
        mean_aupr = mean(aupr_test_list)

        print("--------------------------------------------------------------")
        print("--------------------------------------------------------------")
        print("Round {:02d} | ROC_AUC {:.4f} | AUPR {:.4f} | F1_Score {:.4f}"
              .format(r, mean_roc, mean_aupr, mean_f1))
        print("--------------------------------------------------------------")
        print("--------------------------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
