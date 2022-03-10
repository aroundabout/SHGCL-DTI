import numpy
import numpy as np
from sklearn.decomposition import PCA

estimator1 = PCA(n_components=128)
estimator2 = PCA(n_components=128)

protein_feature_420 = np.loadtxt("../../data/feature/protein_feature_420.txt")
drug_feature_167 = np.loadtxt("../../data/feature/drug_feature_167.txt")

protein_feature = estimator1.fit_transform(protein_feature_420)
drug_feature = estimator2.fit_transform(drug_feature_167)
print(protein_feature)
print(drug_feature)
c = 1
numpy.savetxt("../../data/feature/protein_feature.txt", protein_feature)
numpy.savetxt("../../data/feature/drug_feature.txt", drug_feature)
