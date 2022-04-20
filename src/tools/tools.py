from random import random

import numpy
import numpy as np
import dgl
from dgl import convert
from dgl.heterograph import DGLHeteroGraph
import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from scipy.sparse import coo_matrix
from src.tools.args import parse_args, parse_argsCO
import torch as th

args = parse_argsCO()

device = args.device

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

# 在异构转同构的过程中,num_nodes_per_ntype为[5603,708,1512,4192] 分别对应di drug pro se
drug_len = 708
protein_len = 1512
sideeffect_len = 4192
disease_len = 5603


def get_meta_path():
    return [[DR_PR_I, PR_DR_I], [DR_PR_A, PR_DR_A],
            [DR_DI_A, DI_PR_A]]


def saveTxt(features: list[str], path: str):
    with open(path, "w") as f:
        for feature in features:
            f.write(feature + '\n')
    print("txt save finished")


def load_data():
    network_path = '../../data/data/'

    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    true_drug = drug_len
    drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
    drug_chemical = drug_chemical[:true_drug, :true_drug]
    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')

    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt')
    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')

    num_drug = len(drug_drug)
    num_protein = len(protein_protein)

    # Removed the self-loop
    drug_chemical = drug_chemical - np.identity(num_drug)
    protein_sequence = protein_sequence / 100.
    protein_sequence = protein_sequence - np.identity(num_protein)

    drug_protein = np.loadtxt(network_path + 'mat_drug_protein.txt')

    return drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, \
           protein_disease, drug_protein


#
def splitGraph(g, offset=0):
    train_len = 0.9
    test_len = 0.1
    numEdges = int(g.num_edges(DR_PR_I))
    test_range = (int(offset * numEdges * test_len), int((offset + 1) * numEdges * test_len))
    x = np.ones(numEdges, dtype=bool)
    x[test_range[0]:test_range[1]] = False
    g.edges[DR_PR_I].data['train_mask'] = torch.from_numpy(x)
    test_mask = ~x
    g.edges[DR_PR_I].data['test_mask'] = torch.from_numpy(test_mask)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_feature():
    disease_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/disease_feature.txt")).to(torch.float32).to(device)
    drug_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/drug_feature.txt")).to(torch.float32).to(device)
    protein_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/protein_feature.txt")).to(torch.float32).to(device)
    sideeffect_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/sideeffect_feature.txt")).to(torch.float32).to(device)
    node_features = {drug: drug_feats, protein: protein_feats, disease: disease_feats,
                     sideeffect: sideeffect_feats}
    return node_features


def ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                   protein_disease, drug_protein, args=None, CO=False) -> DGLHeteroGraph:
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = len(drug_disease.T)
    num_sideeffect = len(drug_sideeffect.T)

    list_drug = [(i, i) for i in range(num_drug)]
    list_protein = [(i, i) for i in range(num_protein)]
    list_sideeffect = [(i, i) for i in range(num_sideeffect)]
    list_disease = [(i, i) for i in range(num_disease)]

    drug_chemical = np.array(drug_chemical)
    drug_chemical[drug_chemical < 0.4] = 0
    drug_drug = coo_matrix(drug_drug + drug_chemical)
    # drug_drug = coo_matrix(drug_drug)
    list_DDI = (drug_drug.row, drug_drug.col)

    protein_sequence = np.array(protein_sequence)
    protein_sequence[protein_sequence < 0.6] = 0
    protein_protein = coo_matrix(protein_protein + protein_sequence)
    # protein_protein = coo_matrix(protein_protein)
    list_PPI = (protein_protein.row, protein_protein.col)

    list_SESEI = []

    list_DIDII = []

    drug_protein = coo_matrix(drug_protein)
    list_drug_protein = (drug_protein.row, drug_protein.col)
    list_protein_drug = (drug_protein.col, drug_protein.row)

    drug_sideeffect = coo_matrix(drug_sideeffect)
    list_drug_sideeffect = (drug_sideeffect.row, drug_sideeffect.col)
    list_sideeffect_drug = (drug_sideeffect.col, drug_sideeffect.row)

    drug_disease = coo_matrix(drug_disease)
    list_drug_disease = (drug_disease.row, drug_disease.col)
    list_disease_drug = (drug_disease.col, drug_disease.row)

    protein_disease = coo_matrix(protein_disease)
    list_protein_disease = (protein_disease.row, protein_disease.col)
    list_disease_protein = (protein_disease.col, protein_disease.row)

    list_protein_sideeffect = []
    list_sideeffect_protein = []

    list_disease_sideeffect = []
    list_sideeffect_disease = []

    list_drug_protein_a = []
    list_protein_drug_a = []

    g_HIN = dgl.heterograph({(disease, DI_DI_V, disease): list_disease,
                             (drug, DR_DR_V, drug): list_drug,
                             (protein, PR_PR_V, protein): list_protein,
                             (sideeffect, SE_SE_V, sideeffect): list_sideeffect,

                             (drug, DR_DR_A, drug): list_DDI,
                             (drug, DR_PR_I, protein): list_drug_protein,
                             (drug, DR_PR_A, protein): list_drug_protein_a,
                             (drug, DR_SE_A, sideeffect): list_drug_sideeffect,
                             (drug, DR_DI_A, disease): list_drug_disease,

                             (protein, PR_DR_I, drug): list_protein_drug,
                             (protein, PR_DR_A, drug): list_protein_drug_a,
                             (protein, PR_PR_A, protein): list_PPI,
                             (protein, PR_SE_A, sideeffect): list_protein_sideeffect,
                             (protein, PR_DI_A, disease): list_protein_disease,

                             (sideeffect, SE_DR_A, drug): list_sideeffect_drug,
                             (sideeffect, SE_PR_A, protein): list_sideeffect_protein,
                             (sideeffect, SE_SE_A, sideeffect): list_SESEI,
                             (sideeffect, SE_DI_A, disease): list_sideeffect_disease,

                             (disease, DI_DR_A, drug): list_disease_drug,
                             (disease, DI_PR_A, protein): list_disease_protein,
                             (disease, DI_SE_A, sideeffect): list_disease_sideeffect,
                             (disease, DI_DI_A, disease): list_DIDII,
                             })
    if CO:
        g = g_HIN.edge_type_subgraph([DR_DR_A, DR_PR_I, DR_SE_A, DR_DI_A,
                                      PR_DR_I, PR_PR_A, PR_DI_A,
                                      SE_DR_A,
                                      DI_DR_A, DI_PR_A])
    else:
        g = g_HIN.edge_type_subgraph([DR_DR_A, DR_PR_I, DR_PR_A, DR_SE_A, DR_DI_A,
                                      PR_PR_A, PR_DR_I, PR_DR_A, PR_SE_A, PR_DI_A,
                                      SE_DR_A, SE_PR_A, SE_DI_A, SE_SE_A,
                                      DI_DR_A, DI_PR_A, DI_DI_A, DI_SE_A
                                      ])
    return g


def numConvert(num: int) -> str:
    nid = num
    # 在异构转同构的过程中,num_nodes_per_ntype为[5603,708,1512,4192] 分别对应di drug pro se
    if num < disease_len:
        nid = "DI" + str(num)
    elif disease_len <= num < disease_len + drug_len:
        nid = "DR" + str(num - disease_len)
    elif disease_len + drug_len <= num < disease_len + drug_len + protein_len:
        nid = "PR" + str(num - disease_len - drug_len)
    elif disease_len + drug_len + protein_len <= num:
        nid = "SE" + str(num - disease_len - drug_len - protein_len)
    return nid


def getRandomWalkTrace(g: DGLHeteroGraph, args):
    g_homo = dgl.convert.to_homogeneous(g)
    # traces = dgl.sampling.node2vec_random_walk(g_homo, g_homo.nodes().numpy().tolist(), 1, 1,
    #                                            walk_length=args.walk_length)
    traces = dgl.sampling.random_walk(g_homo, g_homo.nodes().numpy().tolist(),
                                      length=args.walk_length, restart_prob=args.alpha)[0]
    traces = traces.numpy().tolist()

    rwDict = {'DR': {'DR': ([], []), 'PR': ([], []), 'DI': ([], []), 'SE': ([], [])},
              'PR': {'DR': ([], []), 'PR': ([], []), 'DI': ([], []), 'SE': ([], [])},
              'DI': {'DR': ([], []), 'PR': ([], []), 'DI': ([], []), 'SE': ([], [])},
              'SE': {'DR': ([], []), 'PR': ([], []), 'DI': ([], []), 'SE': ([], [])}}
    edgeNameDict = {'DR': {'DR': (drug, DR_DR_A, drug),
                           'PR': (drug, DR_PR_A, protein),
                           'DI': (drug, DR_DI_A, disease),
                           'SE': (drug, DR_SE_A, sideeffect)},
                    'PR': {'DR': (protein, PR_DR_A, drug),
                           'PR': (protein, PR_PR_A, protein),
                           'DI': (protein, PR_DI_A, disease),
                           'SE': (protein, PR_SE_A, sideeffect)},
                    'DI': {'DR': (disease, DI_DR_A, drug),
                           'PR': (disease, DI_PR_A, protein),
                           'DI': (disease, DI_DI_A, disease),
                           'SE': (disease, DI_SE_A, sideeffect)},
                    'SE': {'DR': (sideeffect, SE_DR_A, drug),
                           'PR': (sideeffect, SE_PR_A, protein),
                           'DI': (sideeffect, SE_DI_A, disease),
                           'SE': (sideeffect, SE_SE_A, sideeffect)}}
    for trace in traces:
        src = numConvert(trace[0])
        for index, num in enumerate(trace):
            if index == 0:
                continue
            dst = numConvert(num)
            if int(dst[2:]) == -1:
                # 对于孤立节点来说,他和自己链接不会导致特征的变化
                # if index == 1:
                #     rwDict[src[:2]][dst[:2]][0].append(int(src[2:]))
                #     rwDict[src[:2]][dst[:2]][1].append(int(src[2:]))
                break
            rwDict[src[:2]][dst[:2]][0].append(int(src[2:]))
            rwDict[src[:2]][dst[:2]][1].append(int(dst[2:]))
    return rwDict, edgeNameDict


def ConstructGraphWithRW(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                         protein_disease, drug_protein, args):
    g = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                       protein_disease, drug_protein, args)
    rwDict, edgeNameDict = getRandomWalkTrace(g, args)
    for k, firstdict in edgeNameDict.items():
        for seck, edgename in firstdict.items():
            g.add_edges(rwDict[k][seck][0], rwDict[k][seck][1], etype=edgename)
    return g


def ConstructGraphOnlyWithRW(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                             protein_disease, drug_protein, node_features, args):
    g = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                       protein_disease, drug_protein, node_features, args)
    rwDict, edgeNameDict = getRandomWalkTrace(g, args)
    edgeNameDict['DR']['PR'] = (drug, DR_PR_I, protein)
    edgeNameDict['PR']['DR'] = (protein, PR_DR_I, drug)
    edgeNameDict.setdefault('DRA', {'PRA': (drug, DR_PR_A, protein)})
    edgeNameDict.setdefault('PRA', {'DRA': (protein, PR_DR_A, drug)})
    rwDict.setdefault('DRA', {'PRA': ([], [])})
    rwDict.setdefault('PRA', {'DRA': ([], [])})
    graphData = {
        edgeNameDict[i][j]: (rwDict[i][j][0], rwDict[i][j][1]) for i, fi in rwDict.items() for j, sec in fi.items()
    }
    g_rw = dgl.heterograph(graphData)
    g_rw.add_edges([], [], etype=(drug, DR_PR_A, protein))
    g_rw.add_edges([], [], etype=(protein, PR_DR_A, drug))
    return g_rw


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k).cpu()
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    list_drug = [(i, i) for i in range(drug_len)]
    list_protein = [(i, i) for i in range(protein_len)]
    hetero_graph = dgl.heterograph({etype: (neg_src, neg_dst),
                                    (drug, DR_DR_V, drug): list_drug,
                                    (protein, PR_PR_V, protein): list_protein})
    hetero_graph = hetero_graph.edge_type_subgraph([DR_PR_I])
    return hetero_graph


def construct_postive_graph(dti, etype):
    utype, _, vtype = etype
    src, dst = dti
    list_drug = [(i, i) for i in range(drug_len)]
    list_protein = [(i, i) for i in range(protein_len)]
    hetero_graph = dgl.heterograph({etype: (src, dst),
                                    (drug, DR_DR_V, drug): list_drug,
                                    (protein, PR_PR_V, protein): list_protein})
    hetero_graph = hetero_graph.edge_type_subgraph([DR_PR_I])
    return hetero_graph


def predict_target_pair(pos_h, neg_h):
    pre = torch.cat((pos_h, neg_h), 0).reshape(-1, 1).to(device)
    target = torch.cat(
        (torch.ones((len(pos_h), 1), dtype=torch.float), torch.zeros((len(neg_h), 1), dtype=torch.long)), 0).to(device)
    return pre, target


def compute_loss(pre, target, pos_weight=None):
    crossentropyloss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return crossentropyloss(pre, target)


def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def compute_score(pre, target, pos_weight=None):
    loss = compute_loss(pre, target, pos_weight=pos_weight)
    roc_auc = roc_auc_score(target.cpu().detach().numpy(), pre.cpu().detach().numpy())
    aupr = average_precision_score(target.cpu().detach().numpy(), pre.cpu().detach().numpy())
    return loss, roc_auc, aupr


def shuffle(pos_h, neg_h):
    h = torch.cat((pos_h, neg_h), 0).cpu()
    label = torch.cat(
        (torch.ones((len(pos_h), 1), dtype=torch.float), torch.zeros((len(neg_h), 1), dtype=torch.long)), 0)
    # 打乱h,label
    h = h.detach().numpy()
    label = label.detach().numpy()

    p = numpy.column_stack((h, label))
    numpy.random.shuffle(p)

    h = torch.from_numpy(p[:, :-1]).to(device)
    label = torch.from_numpy(p[:, -1:]).to(device)
    return h, label

def concat_link_pos(graph, feat_src, feat_dst, etype):
    def concat_message_function(edges):
        return {'cat_feat': torch.cat([edges.src['feature'], edges.dst['feature']], 1)}

    with graph.local_scope():
        graph.nodes['drug'].data['feature'] = feat_src
        graph.nodes['protein'].data['feature'] = feat_dst
        graph.apply_edges(concat_message_function, etype=etype)
        pos_h = graph.edata.pop('cat_feat')

    return pos_h

def concat_link(graph, neg_graph, feat_src, feat_dst, etype):
    def concat_message_function(edges):
        return {'cat_feat': torch.cat([edges.src['feature'], edges.dst['feature']], 1)}

    with graph.local_scope():
        graph.nodes['drug'].data['feature'] = feat_src
        graph.nodes['protein'].data['feature'] = feat_dst
        graph.apply_edges(concat_message_function, etype=etype)
        pos_h = graph.edata.pop('cat_feat')
    with neg_graph.local_scope():
        neg_graph.nodes['drug'].data['feature'] = feat_src
        neg_graph.nodes['protein'].data['feature'] = feat_dst
        neg_graph.apply_edges(concat_message_function, etype=etype)
        neg_h = neg_graph.edata.pop('cat_feat')

    return pos_h, neg_h


def evaluate(model, test_graph, test_neg_graph, drug_feature, protein_feature, etype, pos_weight=None):
    with torch.no_grad():
        def concat_message_function(edges):
            return {'cat_feat': torch.cat([edges.src['feature'], edges.dst['feature']], 1)}

        with test_graph.local_scope():
            test_graph.nodes['drug'].data['feature'] = drug_feature
            test_graph.nodes['protein'].data['feature'] = protein_feature
            test_graph.apply_edges(concat_message_function, etype=etype)
            pos_h = test_graph.edata.pop('cat_feat')
        with test_neg_graph.local_scope():
            test_neg_graph.nodes['drug'].data['feature'] = drug_feature
            test_neg_graph.nodes['protein'].data['feature'] = protein_feature
            test_neg_graph.apply_edges(concat_message_function, etype=etype)
            neg_h = test_neg_graph.edata.pop('cat_feat')
        # 写法1
        pre, target = predict_target_pair(model.pred(pos_h), model.pred(neg_h))

        # 写法2
        # h, target = shuffle(pos_h, neg_h)
        # pre = model.pred(h)

        loss, roc_auc, aupr = compute_score(pre, target, pos_weight=pos_weight)
        pre_f1, target_f1 = torch.round(torch.sigmoid(pre.cpu())).detach().numpy(), \
                            torch.round(torch.sigmoid(target.cpu())).detach().numpy()
        f1 = f1_score(target_f1, pre_f1)
        return roc_auc, aupr, f1, loss


def l2_norm(t, axit=1):
    t = t.float()
    norm = th.norm(t, 2, axit, True) + 1e-12
    output = th.div(t, norm)
    output[th.isnan(output) | th.isinf(output)] = 0.0
    return output


def row_normalize(t):
    t = t.float()
    row_sums = t.sum(1) + 1e-12
    output = t / row_sums[:, None]
    output[th.isnan(output) | th.isinf(output)] = 0.0
    return output


if __name__ == "__main__":
    drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, protein_disease, dti_original = load_data()

    # 构建异质图

    hetero_graph = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                  protein_sequence,
                                  protein_disease, dti_original, args)

    g = dgl.metapath_reachable_graph(hetero_graph, [PR_DI_A, DI_PR_A])
    g1 = dgl.metapath_reachable_graph(hetero_graph, [DR_DI_A, DI_DR_A])

    A = 1
