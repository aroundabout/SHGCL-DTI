import dgl
import gensim.models
from collections import OrderedDict
import networkx as nx
from node2vec import Node2Vec
# import sys
# sys.path.append("../../src")

from tools.tools import load_data, ConstructGraph, saveTxt


def getWord2vec():
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
    traces_str = [list(map(str, walk)) for walk in traces]
    print(traces_str)
    skip_gram_params = dict()
    if 'workers' not in skip_gram_params:
        skip_gram_params['workers'] = 6
    if 'size' not in skip_gram_params:
        skip_gram_params['vector_size'] = 128
    skip_gram_params['min_count'] = 1
    print("start fit")
    model = gensim.models.Word2Vec(traces_str, **skip_gram_params)

    # use node2vec to init the se and di's embedding
    # n2v = Node2Vec(g, dimensions=128, walk_length=20, workers=6)
    # model = n2v.fit()
    model.wv.save_word2vec_format("../../data/feature/word2vec_feature.bin")


def sortWord2Vec():
    f = open("../../data/feature/word2vec_feature.bin", "r")
    lines = f.readlines()
    word2vecDict = OrderedDict()
    for index, line in enumerate(lines):
        if index == 0:
            continue
        splitLine = line.split(" ", 1)
        word2vecDict[int(splitLine[0])] = splitLine[1]

    word2vecDict = OrderedDict(sorted(word2vecDict.items(), key=lambda obj: obj[0]))
    allNodesNode2vecFeature = []
    for index, item in enumerate(word2vecDict.items()):
        # 在原本的随机游走采样结果中存在-1
        # Note that if a random walk stops in advance, DGL pads the trace with -1 to have the same length.
        # 所以排除-1行,根据排序结果,忽略第一行
        if index == 0:
            continue
        allNodesNode2vecFeature.append(item[1].strip("\n"))
    print(word2vecDict.keys())
    saveTxt(allNodesNode2vecFeature, "../../data/feature/node2vec_feature.txt")


def test():
    a = 1


if __name__ == "__main__":
    sortWord2Vec()
