# et将xml作为tree处理，最终转化为csv
import xml.etree.cElementTree as et
import pandas as pd

tree = et.parse("full database.xml")
root = tree.getroot()
# 获取根节点的所有tag值，并且作为列名
ori_li = []
id_to_smiles=[]
for x in root:
    for y in x:
        ori_li.append(y.tag)
        dbid=""
        if 'primary' in y.attrib and "drugbank-id" in y.tag:
            dbid=y.text
            smiles=""
            for a in y:
                if "experimental-properties" in a:
                    for b in a:
                        
# 元祖出去重复
ori_set = set(ori_li)
print(ori_li)
dict_ori = {}
for i in ori_set:
    dict_ori[i] = []

# 获得字典
for x in root:
    for y in x:
        if dict_ori[y.tag]:
            dict_ori[y.tag].append(y.text)
        else:
            dict_ori[y.tag].append('nothing')

dict_sec = {}
for i in dict_ori.keys():
    if len(dict_ori[i]) == 14315:
        dict_sec[i] = dict_ori[i]

df = pd.DataFrame.from_dict(dict_sec)

# 去除大多数的无用列名
for i in df.columns:
    if len(df[i].unique()) < 5:
        df.drop(i, axis=1, level=None, inplace=True, errors='raise')
# 重命名列名
for i in df.columns:
    df.rename(columns={i: i.replace('{http://www.drugbank.ca}', '')}, inplace=True)
# 输出为csv
df.to_csv('drugbank.csv', index=False)
import numpy as np

np.save("id_to_smiles.npz",id_to_smiles)