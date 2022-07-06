import sys
import urllib.request
import collections
import urllib
import math
import pandas as pd
import re
import datetime
from AutoCor import CalculateAutoTotal

# some code from pydpi 1.0 because it's written in py2, i rewrite part of it

AALetter = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

_Hydrophobicity = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74, "G": 0.48,
                   "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18,
                   "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}

_hydrophilicity = {"A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0, "G": 0.0, "H": -0.5,
                   "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4,
                   "Y": -2.3, "V": -1.5}

_residuemass = {"A": 15.0, "R": 101.0, "N": 58.0, "D": 59.0, "C": 47.0, "Q": 72.0, "E": 73.0, "G": 1.000, "H": 82.0,
                "I": 57.0, "L": 57.0, "K": 73.0, "M": 75.0, "F": 91.0, "P": 42.0, "S": 31.0, "T": 45.0, "W": 130.0,
                "Y": 107.0, "V": 43.0}

_pK1 = {"A": 2.35, "C": 1.71, "D": 1.88, "E": 2.19, "F": 2.58, "G": 2.34, "H": 1.78, "I": 2.32, "K": 2.20, "L": 2.36,
        "M": 2.28, "N": 2.18, "P": 1.99, "Q": 2.17, "R": 2.18, "S": 2.21, "T": 2.15, "V": 2.29, "W": 2.38, "Y": 2.20}

_pK2 = {"A": 9.87, "C": 10.78, "D": 9.60, "E": 9.67, "F": 9.24, "G": 9.60, "H": 8.97, "I": 9.76, "K": 8.90, "L": 9.60,
        "M": 9.21, "N": 9.09, "P": 10.6, "Q": 9.13, "R": 9.09, "S": 9.15, "T": 9.12, "V": 9.74, "W": 9.39, "Y": 9.11}

_pI = {"A": 6.11, "C": 5.02, "D": 2.98, "E": 3.08, "F": 5.91, "G": 6.06, "H": 7.64, "I": 6.04, "K": 9.47, "L": 6.04,
       "M": 5.74, "N": 10.76, "P": 6.30, "Q": 5.65, "R": 10.76, "S": 5.68, "T": 5.60, "V": 6.02, "W": 5.88, "Y": 5.63}


def GetProteinSequence(ProteinID):
    """
    #########################################################################################
    Get the protein sequence from the uniprot website by ID.

    Usage:

    result=GetProteinSequence(ProteinID)

    Input: ProteinID is a string indicating ID such as "P48039".

    Output: result is a protein sequence.
    #########################################################################################
    """

    ID = str(ProteinID)
    localfile = urllib.request.urlopen('http://www.uniprot.org/uniprot/' + ID + '.fasta')
    temp = localfile.readlines()
    res = ''
    for i in range(1, len(temp)):
        res = res + str(temp[i]).strip("b\'").strip("\\n\'")
    return res


def getProteinSequence(UniportId: str) -> str:
    return GetProteinSequence(UniportId.strip())


def Getkmers():
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers


def CalculateTripeptideComposition(proteinsequence):
    result = {}
    kmers = Getkmers()
    for i in kmers:
        result[i] = len(re.findall(i, proteinsequence))
    return result


# 二肽
def CalculateDipeptideComposition(ProteinSequence):
    # 400
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        for j in AALetter:
            Dipeptide = i + j
            Result[Dipeptide] = round(float(ProteinSequence.count(Dipeptide)) / (LengthSequence - 1) * 100, 2)
    return Result


# 氨基酸
def CalculateAAComposition(ProteinSequence):
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round(float(ProteinSequence.count(i)) / LengthSequence * 100, 3)
    return Result


def getProteinFeatureAuto(sequence: str) -> str:
    temp = CalculateAutoTotal(sequence)
    di = collections.OrderedDict(temp)
    ans = ""
    for item in di.values():
        ans += str(item) + " "
    return ans


def getProteinFeature(sequence: str) -> str:
    dict = {**CalculateAAComposition(sequence), **CalculateDipeptideComposition(sequence),
            **CalculateTripeptideComposition(sequence)}
    dict = collections.OrderedDict(dict)
    ans = ""
    for item in dict.values():
        ans += str(item) + " "
    return ans


def loadProteinUniId(path="../../data/data/protein_dict_map.txt") -> list[str]:
    file = open(path, "r")
    lines = file.readlines()
    uniIds = []
    for line in lines:
        line = line.strip("\n")
        uniIds.append(line.split(":")[0])
    return uniIds


def saveTxt(features, path="../../data/feature/protein_feature_lost.txt"):
    with open(path, "w") as f:
        for feature in features:
            f.write(feature + '\n')
    print("feature txt save finished")


def process():
    feats = []
    columns = ["id", "uniportId", "sequence"]
    proteinSeqs = pd.DataFrame(columns=columns)
    print("now start...")
    print("---------------------------------------------")
    for i, uniId in enumerate(loadProteinUniId()):
        try:
            print("now, the rate of progress is: " + str(i) + "/3837")
            seq = getProteinSequence(uniId)
            proteinSeq = [{"id": i, "uniportId": uniId, "sequence": seq}]
            feat = getProteinFeature(seq)
            proteinSeqs = proteinSeqs.append(proteinSeq)
            feats.append(feat)
        except:
            print("Unexpect error in index", i)
            print("Unexpect", sys.exc_info()[0])
            proteinSeq = [{"id": i, "uniportId": uniId, "sequence": None}]
            proteinSeqs = proteinSeqs.append(proteinSeq)
            feats.append(uniId)

    proteinSeqs.to_csv("../../data/feature/protein_sequence.csv", index=False)
    saveTxt(feats)

    print("done")
    print("---------------------------------------------")


def test():
    timestamp = datetime.datetime.now()
    uid = "P35626"
    seq = getProteinSequence(uid)
    pseq = [{"id": 0, "uniportId": uid, "sequence": seq}]
    feat = getProteinFeature(seq)
    print(pseq)
    print(feat)
    print(datetime.datetime.now() - timestamp)
    timestamp = datetime.datetime.now()


def redirectIdReFetch():
    file = open("../../data/feature/protein_feature_full.txt", "r")
    lines = file.readlines()
    features = []
    lostProtein = []
    index = 0
    for line in lines:
        print("now is: " + str(index) + "/3837")
        index += 1
        feature = line.strip("\n")
        if line[0].isalpha():
            lostProtein.append(line.strip("\n"))
        features.append(feature)
    saveTxt(lostProtein, path="../../data/feature/lostProtein.txt")


def setRedirectProtein():
    file = open("../../data/feature/RedirectMap.txt")
    lines = file.readlines()
    seqOrUniId = {}
    index = 0
    for line in lines:
        print(index)
        index += 1
        line = line.strip("\n")
        items = line.split(" ")
        seqOrUniId[items[0].strip()] = items[1].strip()
    print("dict finish")
    filePF = open("../../data/feature/protein_feature_full.txt", "r")
    linesPF = filePF.readlines()
    feature = []
    index = 0
    for line in linesPF:
        print(index)
        index += 1
        line = line.strip("\n")
        if line[0].isalpha():
            sou = seqOrUniId[line]
            seq = ""
            if sou.isalpha():
                seq = sou
            else:
                seq = getProteinSequence(sou)
            feat = getProteinFeature(seq)
            feature.append(feat)
        else:
            feature.append(line)
    saveTxt(feature, path="../../data/feature/protein_feature_full.txt")


def getProteinFinalFeature():
    # 1.加载csv和txt构建字典
    # 2.protein.txt读取key
    # 3.生成feature
    seqs = dict()

    df = pd.read_csv("../../data/feature/protein_sequence.csv")
    for index, row in df.iterrows():
        seqs[row["uniportId"]] = row["sequence"]
    f = open("../../data/feature/RedirectMap.txt", "r")
    lines = f.readlines()
    for line in lines:
        kv = line.strip("\n").split(" ")
        seqs[kv[0]] = kv[1]
    pf = open("../../data/data/protein.txt")
    proteins = pf.readlines()
    protein_feature = []
    lostProtein = []
    redirectMap2 = []
    for protein in proteins:
        protein = protein.strip("\n")
        # if protein not in seqs.keys():
        #     try:
        #         tempseq = GetProteinSequence(protein)
        #         redirectMap2.append(protein + " " + tempseq)
        #         print(protein+" "+tempseq)
        #         continue
        #     except:
        #         print("error, 300 ", protein)
        #         lostProtein.append(protein)
        #         continue
        seq = seqs[protein]
        if not seq.isalpha():
            seq = getProteinSequence(seq)
        protein_feature.append(getProteinFeature(seq))
        # protein_feature.append(getProteinFeatureAuto(seq))

    # saveTxt(protein_feature, "../../data/feature/protein_feature_420.txt")
    saveTxt(protein_feature, "../../data/feature/protein_feature_8420.txt")
    # saveTxt(redirectMap2, "../../data/feature/RedirectMap2.txt")


if __name__ == "__main__":
    # test()
    getProteinFinalFeature()
