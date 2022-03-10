import pandas as pd
from rdkit import Chem
from base64 import b64decode
from rdkit.Chem import MACCSkeys
import numpy as np


# for index, row in df.iterrows():
#     if index % 100 == 0:
#         print(index)
#         print("heihei ")
#     drugBankId = row["drugbank_id"]
#     if index_dbids==len(dbids):
#         break
#     if drugBankId == dbids[index_dbids]:
#         index_dbids += 1
#         smiles = row["smiles"]
#         mol = Chem.MolFromSmiles(smiles)
#         feat = ""
#         if mol is None:
#             feat = "0 " * 166 + "0"
#         else:
#             numFinger, res, bv, feat = CalculateMACCSFingerprint(mol)
#         drugFeatures.append(feat)
#     elif drugBankId > dbids[index_dbids]:
#         index_dbids += 1
#     else:
#         continue

def PCFP_BitString(pcfp_base64):
    pcfp_bitstring = "".join(["{:08b}".format(x) for x in b64decode(pcfp_base64)])[32:913]
    return pcfp_bitstring


def saveTxt(features: list, path="../../data/feature/drug_feature.txt"):
    with open(path, "w") as f:
        for feature in features:
            f.write(feature + '\n')
    print("feature txt save finished")


def CalculateMACCSFingerprint(mol):
    """
    Calculate MACCS keys (166 bits).
    """
    res = {}
    NumFinger = 166
    bv = MACCSkeys.GenMACCSKeys(mol)
    temp = tuple(bv.GetOnBits())
    featList = [0] * 167
    feat = ""
    for i in temp:
        res.update({i: 1})
        featList[i] = 1
    for item in featList:
        feat += str(item) + " "
    return NumFinger, res, bv, feat


def loadDrugBankId(path="../../data/data/drug.txt") -> list[str]:
    file = open(path, "r")
    lines = file.readlines()
    dbIds = []
    for line in lines:
        line = line.strip("\n")
        dbIds.append(line)
    return dbIds


# 这里随机不是很关键 因为到时候会在同意做标准化减少单个节点的影响
def random_feature() -> str:
    fList = np.random.randn(167)
    ans = ""
    for f in fList:
        ans += str(f) + " "
    return ans


if __name__ == "__main__":
    drugFeatures = []
    lostDrug = []
    df = pd.read_csv("../../data/feature/preprocessData/drugbank_smiles.csv")
    dbids = loadDrugBankId()
    index_dbids = 0

    for index, row in enumerate(dbids):
        print(index)
        value_result = df.loc[df["drugbank_id"] == row].index.tolist()
        feat = ""
        if len(value_result) == 0:
            feat = random_feature()
        else:
            datarow = df.iloc[value_result[0]]
            smiles = datarow["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                feat = random_feature()
            else:
                numFinger, res, bv, feat = CalculateMACCSFingerprint(mol)
        drugFeatures.append(feat)

    saveTxt(drugFeatures)
    saveTxt(lostDrug, path="../../data/feature/lost_drug.txt")
