import dgl
import gensim.models

from tools.tools import load_data, ConstructGraph, saveTxt


def numConvert(num: int) -> str:
    nid = num
    # 在异构转同构的过程中,num_nodes_per_ntype为[5603,708,1512,4192] 分别对应di drug pro se
    if num < 5603:
        nid = "DI" + str(num)
    elif 5603 <= num < 5603 + 708:
        nid = "DR" + str(num)
    elif 5603 + 708 <= num < 5603 + 708 + 1512:
        nid = "PR" + str(num)
    elif 5603 + 708 + 1512 <= num:
        nid = "SE" + str(num)
    return nid


def rwr():
    drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, protein_disease, dti_original = load_data()

    # 构建异质图
    g = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                       protein_disease, dti_original)
    # 转化为同构图
    g_homo = dgl.convert.to_homogeneous(g)
    # g = dgl.convert.to_networkx(g)
    print("convert finish")
    print(g_homo.nodes())
    traces = dgl.sampling.node2vec_random_walk(g_homo, g_homo.nodes().numpy().tolist(), 1, 1, walk_length=40)
    traces = traces.numpy().tolist()

    rwDict = {'DR': {'DR': [],
                     'PR': [],
                     'DI': [],
                     'SE': []},
              'PR': {'DR': [],
                     'PR': [],
                     'DI': [],
                     'SE': []},
              'DI': {'DR': [],
                     'PR': [],
                     'DI': [],
                     'SE': []},
              'SE': {'DR': [],
                     'PR': [],
                     'DI': [],
                     'SE': []}}

    for trace in traces:
        src = numConvert(trace[0])
        for index, num in enumerate(trace):
            if index == 0:
                continue
            dst = numConvert(num)
            rwDict[src[:2]][dst[:2]].append(int(src[2:]), int[dst[2:]])
    return rwDict


rwr()
