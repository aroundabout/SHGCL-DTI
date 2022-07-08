import numpy as np
import scipy.sparse as sp

from tools.tools import sparse_mx_to_torch_sparse_tensor, normalize_adj


def get_mp(drug_drug, durg_protein, drug_disease, drug_se, protein_protein, protein_disease, device):
    drdr = drug_drug
    drpr = durg_protein
    prpr = protein_protein
    prdi = protein_disease
    drse = drug_se
    drdi = drug_disease

    drdrdr = np.matmul(drdr.T, drdr) > 0
    drprdr = np.matmul(drpr, drpr.T) > 0
    drdrprdr = np.matmul(drdr, drprdr.T) > 0
    drsedr = np.matmul(drse, drse.T) > 0
    drdidr = np.matmul(drdi, drdi.T) > 0
    prdrpr = np.matmul(drpr.T, drpr) > 0
    prprpr = np.matmul(prpr.T, prpr) > 0
    prdipr = np.matmul(prdi, prdi.T) > 0
    drprpr = np.matmul(drpr, prpr) > 0
    prprdrpr = np.matmul(prpr, prdrpr.T) > 0
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

    drdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drdr))).to(device)
    drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprdr))).to(device)
    drdrdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drdrdr))).to(device)
    drprprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprprdr))).to(device)
    drprdrprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprdrprdr))).to(device)
    drprdiprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprdiprdr))).to(device)
    drdrprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drdrprdr))).to(device)

    prpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prpr))).to(device)
    prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrpr))).to(device)
    prprpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prprpr))).to(device)
    prdrprdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrprdrpr))).to(device)
    prprprpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prprprpr))).to(device)
    prdrdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrdrpr))).to(device)
    prprdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prprdrpr))).to(device)

    drug = 'drug'
    protein = 'protein'
    disease = 'disease'
    sideeffect = 'sideeffect'

    mps_dict = {drug: [drdrdr, drprdr, drprprdr, drprdrprdr, drprdiprdr],
                protein: [prdrpr, prprpr, prdrprdrpr, prprprpr, prdrdrpr], disease: [], sideeffect: []}
    mps_dict = {drug: [drprdr], protein: [prdrpr], disease: [], sideeffect: []}
    # mps_dict = {drug: [drprdr, drprdrprdr], protein: [prdrpr, prdrprdrpr], disease: [], sideeffect: []}
    # mps_dict = {drug: [drdr, drprdr, drprprdr], protein: [prpr, prdrpr, prdrdrpr], disease: [], sideeffect: []}
    # mps_dict = {drug: [drprdr], protein: [prdrpr], disease: [], sideeffect: []}

    return mps_dict
