import numpy
import numpy as np
import scipy.sparse as sp
from collections import Counter
from src.tools.tools import load_data, ConstructGraph
from src.tools.args import parse_args
from scipy.sparse import coo_matrix

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


def getMetaPathSrcAndDst(g, metapath):
    adj = 1
    for etype in metapath:
        adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=False)

    adj = adj.tocoo()
    return adj


drug_pos_num = 20
drug_num = 708
protein_pos_num = 40
protein_num = 1512
sideeffect_pos_num = 40
sideeffect_num = 4192
disease_pos_num = 40
disease_num = 5603

drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, \
protein_sequence, protein_disease, dti_original = load_data()

# 构建异质图
args = parse_args()

g = ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                   protein_sequence, protein_disease, dti_original, args)

drdrdr = getMetaPathSrcAndDst(g, [DR_DR_A, DR_DR_A])
drprdr = getMetaPathSrcAndDst(g, [DR_PR_I, PR_DR_I])
drsedr = getMetaPathSrcAndDst(g, [DR_SE_A, SE_DR_A])
drdidr = getMetaPathSrcAndDst(g, [DR_DI_A, DI_DR_A])
# drdrdr = drdrdr / drdrdr.sum(axis=-1).reshape(-1, 1)
# drprdr = drprdr / drprdr.sum(axis=-1).reshape(-1, 1)
# drsedr = drsedr / drsedr.sum(axis=-1).reshape(-1, 1)
# drdidr = drdidr / drdidr.sum(axis=-1).reshape(-1, 1)

drug_all = (drdrdr + drprdr + drsedr + drdidr).A.astype("float32")
all_ = (drug_all > 0).sum(-1)
print(all_.max(), all_.min(), all_.mean())
drug_pos = np.zeros((drug_num, drug_num))
for i in range(len(drug_all)):
    one = drug_all[i].nonzero()[0]
    if len(one) > drug_pos_num:
        oo = np.argsort(-drug_all[i, one])
        sele = one[oo[:drug_pos_num]]
        drug_pos[i, sele] = 1
    else:
        drug_pos[i, one] = 1
drug_pos = sp.coo_matrix(drug_pos)
sp.save_npz("../../data/pos/drug_pos.npz", drug_pos)
# protein


prdrpr = getMetaPathSrcAndDst(g, [PR_DR_I, DR_PR_I])
prprpr = getMetaPathSrcAndDst(g, [PR_PR_A, PR_PR_A])
prdipr = getMetaPathSrcAndDst(g, [PR_DI_A, DI_PR_A])
# prdrpr = prdrpr / prdrpr.sum(axis=-1).reshape(-1, 1)
# prprpr = prprpr / prprpr.sum(axis=-1).reshape(-1, 1)
# prdipr = prdipr / prdipr.sum(axis=-1).reshape(-1, 1)
protein_all = (prdrpr + prprpr + prdipr).A.astype("float32")
all_ = (protein_all > 0).sum(-1)
print(all_.max(), all_.min(), all_.mean())
protein_pos = np.zeros((protein_num, protein_num))
for i in range(len(protein_all)):
    one = protein_all[i].nonzero()[0]
    if len(one) > protein_pos_num:
        oo = np.argsort(-protein_all[i, one])
        sele = one[oo[:protein_pos_num]]
        protein_pos[i, sele] = 1
    else:
        protein_pos[i, one] = 1
protein_pos = sp.coo_matrix(protein_pos)
sp.save_npz("../../data/pos/protein_pos.npz", protein_pos)

# sideeffect
sedrse = getMetaPathSrcAndDst(g, [SE_DR_A, DR_SE_A])
# sedrse = sedrse / sedrse.sum(axis=-1).reshape(-1, 1)
se_all = (sedrse).A.astype("float32")
all_ = (se_all > 0).sum(-1)
sideeffect_pos = np.zeros((sideeffect_num, sideeffect_num))
for i in range(len(se_all)):
    one = se_all[i].nonzero()[0]
    if len(one) > sideeffect_pos_num:
        oo = np.argsort(-se_all[i, one])
        sele = one[oo[:sideeffect_pos_num]]
        sideeffect_pos[i, sele] = 1
    else:
        sideeffect_pos[i, one] = 1
sideeffect_pos = sp.coo_matrix(sideeffect_pos)
sp.save_npz("../../data/pos/sideeffect_pos.npz", sideeffect_pos)

# disease
didrdi = getMetaPathSrcAndDst(g, [DI_DR_A, DR_DI_A])
diprdi = getMetaPathSrcAndDst(g, [DI_PR_A, PR_DI_A])
# didrdi = didrdi / didrdi.sum(axis=-1).reshape(-1, 1)
# diprdi = diprdi / diprdi.sum(axis=-1).reshape(-1, 1)
disease_all = (didrdi + diprdi).A.astype("float32")
all_ = (disease_all > 0).sum(-1)
disease_pos = np.zeros((disease_num, disease_num))
for i in range(len(disease_all)):
    one = disease_all[i].nonzero()[0]
    if len(one) > disease_pos_num:
        oo = np.argsort(-disease_all[i, one])
        sele = one[oo[:disease_pos_num]]
        disease_pos[i, sele] = 1
    else:
        disease_pos[i, one] = 1
disease_pos = sp.coo_matrix(disease_pos)
sp.save_npz("../../data/pos/disease_pos.npz", disease_pos)
