import numpy as np
import scipy.sparse as sp

####################################################
# This tool is to generate meta-path based adjacency
# matrix given original links.
####################################################

drug_len = 708
protein_len = 1512
sideeffect_len = 4192
disease_len = 5603
true_drug = drug_len

network_path = '../../data/data/'

drdr = np.loadtxt(network_path + 'mat_drug_drug.txt')
drpr = np.loadtxt(network_path + 'mat_drug_protein.txt')
drdi = np.loadtxt(network_path + 'mat_drug_disease.txt')
drse = np.loadtxt(network_path + 'mat_drug_se.txt')
prpr = np.loadtxt(network_path + 'mat_protein_protein.txt')
prdi = np.loadtxt(network_path + 'mat_protein_disease.txt')

# drdr = sp.coo_matrix(drdr).toarray()
# drpr = sp.coo_matrix(drpr).toarray()
# drdi = sp.coo_matrix(drdi).toarray()
# drse = sp.coo_matrix(drse).toarray()
# prpr = sp.coo_matrix(prpr).toarray()
# prdi = sp.coo_matrix(prdi).toarray()


# drug
drdrdr = np.matmul(drdr.T, drdr) > 0
drdrdr = sp.coo_matrix(drdrdr)
sp.save_npz('../../data/mp/drdrdr.npz', drdrdr)

drprdr = np.matmul(drpr, drpr.T) > 0
drprdr = sp.coo_matrix(drprdr)
sp.save_npz('../../data/mp/drprdr.npz', drprdr)

drdidr = np.matmul(drdi, drdi.T) > 0
drdidr = sp.coo_matrix(drdidr)
sp.save_npz('../../data/mp/drdidr.npz', drdidr)

drsedr = np.matmul(drse, drse.T) > 0
drsedr = sp.coo_matrix(drsedr)
sp.save_npz('../../data/mp/drsedr.npz', drsedr)

drdipr = np.matmul(drdi, prdi.T)
drdiprdidr = np.matmul(drdipr, drdipr.T) > 0
drdiprdidr = sp.coo_matrix(drdiprdidr)
sp.save_npz('../../data/mp/drdiprdidr.npz', drdiprdidr)

# protein
prdrpr = np.matmul(drpr.T, drpr) > 0
prdrpr = sp.coo_matrix(prdrpr)
sp.save_npz('../../data/mp/prdrpr.npz', prdrpr)

prprpr = np.matmul(prpr.T, prpr) > 0
prprpr = sp.coo_matrix(prprpr)
sp.save_npz('../../data/mp/prprpr.npz', prprpr)

prdipr = np.matmul(prdi, prdi.T) > 0
prdipr = sp.coo_matrix(prdipr)
sp.save_npz('../../data/mp/prdipr.npz', prdipr)

prdidr = np.matmul(prdi, drdi.T) > 0
prdidrdipr = np.matmul(prdidr, prdidr.T)
prdidrdipr = sp.coo_matrix(prdidrdipr)
sp.save_npz('../../data/mp/prdidrdipr.npz', prdidrdipr)

# disease
didrdi = np.matmul(drdi.T, drdi) > 0
didrdi = sp.coo_matrix(didrdi)
sp.save_npz('../../data/mp/didrdi.npz', didrdi)

diprdi = np.matmul(prdi.T, prdi) > 0
diprdi = sp.coo_matrix(diprdi)
sp.save_npz('../../data/mp/diprdi.npz', diprdi)

# sideeffect
sedrse = np.matmul(drse.T, drse) > 0
sedrse = sp.coo_matrix(sedrse)
sp.save_npz('../../data/mp/sedrse.npz', sedrse)
