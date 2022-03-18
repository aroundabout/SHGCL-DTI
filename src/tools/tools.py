import numpy
import numpy as np
import dgl
from dgl.heterograph import DGLHeteroGraph
import torch
from dgl.data.utils import generate_mask_tensor
from sklearn.metrics import roc_auc_score, f1_score

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


def saveTxt(features: list[str], path: str):
    with open(path, "w") as f:
        for feature in features:
            f.write(feature + '\n')
    print("feature txt save finished")


def load_data():
    network_path = '../../data/data/'

    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    true_drug = 708
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

    print("Load data finished.")

    return drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, \
           protein_disease, drug_protein


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


def ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                   protein_disease, drug_protein) -> DGLHeteroGraph:
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = len(drug_disease.T)
    num_sideeffect = len(drug_sideeffect.T)

    list_drug = [(i, i) for i in range(num_drug)]
    list_protein = [(i, i) for i in range(num_protein)]
    list_sideeffect = [(i, i) for i in range(num_sideeffect)]
    list_disease = [(i, i) for i in range(num_disease)]

    list_DDI = []
    for row in range(num_drug):
        for col in range(num_drug):
            if drug_drug[row, col] > 0:
                list_DDI.append((row, col))

    list_PPI = []
    for row in range(num_protein):
        for col in range(num_protein):
            if protein_protein[row, col] > 0:
                list_PPI.append((row, col))

    list_SESEI = []

    list_DIDII = []

    list_drug_protein = []
    list_protein_drug = []
    for row in range(num_drug):
        for col in range(num_protein):
            if drug_protein[row, col] > 0:
                list_drug_protein.append((row, col))
                list_protein_drug.append((col, row))

    list_drug_sideeffect = []
    list_sideeffect_drug = []
    for row in range(num_drug):
        for col in range(num_sideeffect):
            if drug_sideeffect[row, col] > 0:
                list_drug_sideeffect.append((row, col))
                list_sideeffect_drug.append((col, row))

    list_drug_disease = []
    list_disease_drug = []
    for row in range(num_drug):
        for col in range(num_disease):
            if drug_disease[row, col] > 0:
                list_drug_disease.append((row, col))
                list_disease_drug.append((col, row))

    list_protein_disease = []
    list_disease_protein = []
    for row in range(num_protein):
        for col in range(num_disease):
            if protein_disease[row, col] > 0:
                list_protein_disease.append((row, col))
                list_disease_protein.append((col, row))

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

    g = g_HIN.edge_type_subgraph([DR_DR_A, DR_PR_I, DR_PR_A, DR_SE_A, DR_DI_A,
                                  PR_PR_A, PR_DR_I, PR_DR_A, PR_SE_A, PR_DI_A,
                                  SE_DR_A, SE_PR_A, SE_DI_A, SE_SE_A,
                                  DI_DR_A, DI_PR_A, DI_DI_A, DI_SE_A
                                  ])
    g.nodes['disease'].data['feature'] = torch.from_numpy(
        numpy.loadtxt("../../data/feature/disease_feature.txt")).to(torch.float32)
    g.nodes['drug'].data['feature'] = torch.from_numpy(
        numpy.loadtxt("../../data/feature/drug_feature.txt")).to(torch.float32)
    g.nodes['protein'].data['feature'] = torch.from_numpy(
        numpy.loadtxt("../../data/feature/protein_feature.txt")).to(torch.float32)
    g.nodes['sideeffect'].data['feature'] = torch.from_numpy(
        numpy.loadtxt("../../data/feature/sideeffect_feature.txt")).to(torch.float32)
    return g


def numConvert(num: int) -> str:
    nid = num
    # 在异构转同构的过程中,num_nodes_per_ntype为[5603,708,1512,4192] 分别对应di drug pro se
    if num < 5603:
        nid = "DI" + str(num)
    elif 5603 <= num < 5603 + 708:
        nid = "DR" + str(num - 5603)
    elif 5603 + 708 <= num < 5603 + 708 + 1512:
        nid = "PR" + str(num - 5603 - 708)
    elif 5603 + 708 + 1512 <= num:
        nid = "SE" + str(num - 5603 - 708 - 1512)
    return nid


def getRandomWalkTrace(g: DGLHeteroGraph):
    g_homo = dgl.convert.to_homogeneous(g)
    print("convert finish")
    print(g_homo.nodes())
    traces = dgl.sampling.node2vec_random_walk(g_homo, g_homo.nodes().numpy().tolist(), 1, 1, walk_length=20)
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
                print("nmd 果然是你", int(src[2:]), int(dst[2:]))
                # 对于孤立节点来说,他和自己链接不会导致特征的变化
                if index == 1:
                    rwDict[src[:2]][dst[:2]][0].append(int(src[2:]))
                    rwDict[src[:2]][dst[:2]][1].append(int(src[2:]))
                break
            rwDict[src[:2]][dst[:2]][0].append(int(src[2:]))
            rwDict[src[:2]][dst[:2]][1].append(int(dst[2:]))
    return rwDict, edgeNameDict


def ConstructGraphWithRW(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                         protein_disease, drug_protein, offset=0):
    g = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                       protein_disease, drug_protein)
    rwDict, edgeNameDict = getRandomWalkTrace(g)
    index = 0
    for k, firstdict in edgeNameDict.items():
        for seck, edgename in firstdict.items():
            g.add_edges(rwDict[k][seck][0], rwDict[k][seck][1], etype=edgename)
    print("heterogarph with random walk finish")
    splitGraph(g, offset=offset)

    return g


def ConstructGraphOnlyWithRW(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                             protein_disease, drug_protein, offset=0):
    g = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                       protein_disease, drug_protein)
    rwDict, edgeNameDict = getRandomWalkTrace(g)
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

    print("heterogarph only with random walk finish")
    splitGraph(g_rw, offset=offset)
    return g_rw


def construct_negative_graph(graph, k, etype, feature):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    list_drug = [(i, i) for i in range(708)]
    list_protein = [(i, i) for i in range(1512)]
    hetero_graph = dgl.heterograph({etype: (neg_src, neg_dst),
                                    (drug, DR_DR_V, drug): list_drug,
                                    (protein, PR_PR_V, protein): list_protein})
    hetero_graph.nodes['drug'].data['feature'] = feature['drug']
    hetero_graph.nodes['protein'].data['feature'] = feature['protein']
    return hetero_graph


def construct_test_graph(dti, etype, feature):
    utype, _, vtype = etype
    src, dst = dti
    list_drug = [(i, i) for i in range(708)]
    list_protein = [(i, i) for i in range(1512)]

    hetero_graph = dgl.heterograph({etype: (src, dst),
                                    (drug, DR_DR_V, drug): list_drug,
                                    (protein, PR_PR_V, protein): list_protein})
    hetero_graph.nodes['drug'].data['feature'] = feature['drug']
    hetero_graph.nodes['protein'].data['feature'] = feature['protein']
    hetero_graph = hetero_graph.edge_type_subgraph([DR_PR_I])
    return hetero_graph


def predict_target_pair(pos_h, neg_h):
    pre = torch.cat((pos_h, neg_h), 0)
    target = torch.cat(
        (torch.ones((len(pos_h), 1), dtype=torch.float), torch.zeros((len(neg_h), 1), dtype=torch.long)), 0)
    return pre, target


def compute_loss(pre, target):
    crossentropyloss = torch.nn.BCEWithLogitsLoss()
    return crossentropyloss(pre, target)


def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, test_graph, test_neg_graph, node_features, etype):
    with torch.no_grad():
        pos_score, neg_score, edge_emb = model(test_graph, test_neg_graph, node_features, etype)
        pre, target = predict_target_pair(pos_score, neg_score)
        roc_auc = roc_auc_score(target.detach().numpy(), pre.detach().numpy())
        pre_f1, target_f1 = torch.round(torch.sigmoid(pre)).detach().numpy(), torch.round(
            torch.sigmoid(target)).detach().numpy()
        f1 = f1_score(target_f1, pre_f1)
        return roc_auc, f1
