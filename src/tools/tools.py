import numpy as np
import dgl


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

    # Removed DTIs with similar drugs or proteins
    # drug_protein = np.loadtxt(network_path + 'mat_drug_protein_homo_protein_drug.txt')

    print("Load data finished.")

    return drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, \
           protein_disease, drug_protein


def ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                   protein_disease, drug_protein):
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = len(drug_disease.T)
    num_sideeffect = len(drug_sideeffect.T)

    list_drug = []
    for i in range(num_drug):
        list_drug.append((i, i))

    list_protein = []
    for i in range(num_protein):
        list_protein.append((i, i))

    list_disease = []
    for i in range(num_disease):
        list_disease.append((i, i))

    list_sideeffect = []
    for i in range(num_sideeffect):
        list_sideeffect.append((i, i))

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

    g_HIN = dgl.heterograph({('disease', 'disease_disease virtual', 'disease'): list_disease,
                             ('drug', 'drug_drug virtual', 'drug'): list_drug,
                             ('protein', 'protein_protein virtual', 'protein'): list_protein,
                             ('sideeffect', 'sideeffect_sideeffect virtual', 'sideeffect'): list_sideeffect,
                             ('drug', 'drug_drug interaction', 'drug'): list_DDI, \
                             ('protein', 'protein_protein interaction', 'protein'): list_PPI, \
                             ('drug', 'drug_protein interaction', 'protein'): list_drug_protein, \
                             ('protein', 'protein_drug interaction', 'drug'): list_protein_drug, \
                             ('drug', 'drug_sideeffect association', 'sideeffect'): list_drug_sideeffect, \
                             ('sideeffect', 'sideeffect_drug association', 'drug'): list_sideeffect_drug, \
                             ('drug', 'drug_disease association', 'disease'): list_drug_disease, \
                             ('disease', 'disease_drug association', 'drug'): list_disease_drug, \
                             ('protein', 'protein_disease association', 'disease'): list_protein_disease, \
                             ('disease', 'disease_protein association', 'protein'): list_disease_protein})

    g = g_HIN.edge_type_subgraph(['drug_drug interaction', 'protein_protein interaction',
                                  'drug_protein interaction', 'protein_drug interaction',
                                  'drug_sideeffect association', 'sideeffect_drug association',
                                  'drug_disease association', 'disease_drug association',
                                  'protein_disease association', 'disease_protein association'
                                  ])

    return g


def saveTxt(features: list[str], path: str):
    with open(path, "w") as f:
        for feature in features:
            f.write(feature + '\n')
    print("feature txt save finished")


if __name__ == "__main__":
    print("tools.py")
