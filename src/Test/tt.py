import dgl
import numpy
import numpy as np
import torch
import torch as th

from tools.tools import ConstructGraphWithRW, load_data
from tools.args import parse_args

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
#
#
# from scipy.sparse import coo_matrix
#
# l = np.array([[2, 0, 0], [0.1, 0, 0], [0.7, 0.5, 1], [0, 0, 1]])
# l[l < 0.5] = 0
# c = coo_matrix(l)
# ans=(c.col,c.row)
#
# print(ans)

src = [1, 1]
dst = [1, 2]
a = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a[src, dst])
