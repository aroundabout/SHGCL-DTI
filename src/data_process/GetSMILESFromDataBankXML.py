import untangle
import pandas as pd
import numpy as np
import os

filename= "../full database.xml"

objective = untangle.parse(filename)

drugbank1 = pd.DataFrame(
    columns=["drugbank_id", "name",  "smiles"])

i = -1

for drug in objective.drugbank.drug:

    i = i + 1


    for id in drug.drugbank_id:
        if str(id["primary"]) == "true":
            drugbank1.loc[i, "drugbank_id"] = id.cdata
       # Drug name
    drugbank1.loc[i, "name"] = drug.name.cdata
    j = 0
   #这里要写一个异常处理，因为并不是所有的药物都有 calculated_properties属性
    try:
        len(drug.calculated_properties.cdata) == 0
    except:
        print('1')
    else:
        if len(drug.calculated_properties.cdata) == 0:
            continue
        else:
            for property in drug.calculated_properties.property:
                if property.kind.cdata == "SMILES":
                    drugbank1.loc[i, "smiles"] = property.value.cdata
#舍弃掉列表中没有smiles的药物

drugbank_smiles = drugbank1.dropna()
drugbank_smiles = drugbank_smiles.reset_index(drop=True)
#写入csv
drugbank_smiles.to_csv("drugbank_smiles.csv", encoding='utf-8', index=False)

