from node2vec import Node2Vec
import networkx as nx
import dgl

G = nx.petersen_graph()

n2v = Node2Vec(G, dimensions=128, walk_length=20, workers=6)
print("start fit")
model = n2v.fit()
model.wv.save_word2vec_format("n2vtest.bin")
