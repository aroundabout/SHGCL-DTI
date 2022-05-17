import numpy as np
import scipy.sparse as sp

from tools.tools import sparse_mx_to_torch_sparse_tensor, normalize_adj


def get_mp(drug_drug, durg_protein, drug_disease, drug_se, protein_protein, protein_disease, device):
    drdr = drug_drug
    drpr = durg_protein
    prpr = protein_protein
    prdi = protein_disease
    drdrdr = np.matmul(drdr.T, drdr) > 0
    drprdr = np.matmul(drpr, drpr.T) > 0
    prdrpr = np.matmul(drpr.T, drpr) > 0
    prprpr = np.matmul(prpr.T, prpr) > 0
    drprpr = np.matmul(drpr, prpr) > 0
    drprprdr = np.matmul(drprpr, drpr.T) > 0
    drprdr = np.array(drprdr)
    drprdrprdr = np.matmul(drprdr.T, drprdr) > 0
    drprdi = np.matmul(drpr, prdi) > 0
    drprdiprdr = np.matmul(drprdi, drprdi.T) > 0
    prdrpr = np.array(prdrpr)
    prdrprdrpr = np.matmul(prdrpr.T, prdrpr) > 0
    prprpr = np.array(prprpr)
    prprprpr = np.matmul(prprpr.T, prpr) > 0
    prdrdr = np.matmul(drpr.T, drdr) > 0
    prdrdrpr = np.matmul(prdrdr, drpr) > 0

    drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprdr))).to(device)
    drdrdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drdrdr))).to(device)
    drprprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprprdr))).to(device)
    drprdrprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprdrprdr))).to(device)
    drprdiprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprdiprdr))).to(device)

    prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrpr))).to(device)
    prprpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prprpr))).to(device)
    prdrprdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrprdrpr))).to(device)
    prprprpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prprprpr))).to(device)
    prdrdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrdrpr))).to(device)

    return drprdr, drdrdr, drprprdr, drprdrprdr, drprdiprdr, prdrpr, prprpr, prdrprdrpr, prprprpr, prdrdrpr
