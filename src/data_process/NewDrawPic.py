import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def double(x_data, y1, y2, c1, name, file_name, l1, h1, l2, h2):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    plt.suptitle(name)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax1)
    y_data = y1
    x_data = x_data
    bar = plt.bar(x_data, y_data, 0.5, color=c1, edgecolor='grey')
    plt.ylabel("AUROC", fontsize=10)
    plt.ylim(l1, h1)
    for a, b in zip(x_data, y_data):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom')
    plt.sca(ax2)
    y_data = y2
    x_data = x_data
    bar = plt.bar(x_data, y_data, 0.5, color=c1, edgecolor='grey')
    plt.ylabel("AUPR", fontsize=10)
    plt.ylim(l2, h2)
    for a, b in zip(x_data, y_data):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom')
    plt.savefig(file_name + '.jpg')
    plt.show()


if __name__ == '__main__':
    testa = '对比实验Test a\npos:neg=1:10'
    testb = '对比实验Test b\nall unknown pairs were treated as negative examples'
    testc = '对比实验Test c\npos:neg=1:10\nDTIs with similar drugs or proteins were removed'
    testd = '对比实验Test d\npos:neg=1:10\nDTIs with drugs sharing similar drug interactions were removed'
    teste = '对比实验Test e\npos:neg=1:10\nDTIs with drugs sharing similar side-effects were removed'
    testf = '对比实验Test f\nTrain on non-unique interactions test on unique interactions'
    testg = '消融实验Test I\nRemove DDI and PPI networks'
    testh = '消融实验Test II\nRemove drugs and protein similarity networks'
    testi = '消融实验Test III\nRemove Side-effect and disease'
    com_exp = ["BLMNII", "MSCMF", "NetLapRLS", "DTINet", 'NeoDTI', 'SHGCL-DTI']
    auc_benchmark = [0.8517, 0.8143, 0.9097, 9.1511252e-01, 9.487602873070016818e-01, 0.96403536899248]
    aupr_benchmark = [0.4682, 0.5755, 0.7532, 8.1858833e-01, 8.554157481527019380e-01, 0.8743130229290721]
    auc_all = [0.8509, 0.8607, 0.8883, 9.0345343e-01, 9.107040247496172025e-01, 0.9281954198208933]
    aupr_all = [0.2098, 0.3846, 0.4070, 4.3416381e-01, 5.820445636266279310e-01, 0.5842407416819303]
    auc_homo = [0.7953, 0.6778, 0.8764, 8.4219046e-01, 8.878660521569964326e-01, 0.9094215497287781]
    aupr_homo = [0.3770, 0.3078, 0.6090, 6.0586271e-01, 6.759711322289558844e-01, 0.73406204850801198]
    auc_drug = [0.8395, 0.8507, 0.9068, 9.0302512e-01, 9.364236222338029325e-01, 0.9476476920845742]
    aupr_drug = [0.4114, 0.6276, 0.7330, 7.5761351e-01, 8.248277060857350795e-01, 0.8395620778015784]
    auc_sideeffect = [0.8525, 0.8491, 0.8939, 9.1439269e-01, 9.465623094573345497e-01, 0.9540500314592931]
    aupr_sideeffect = [0.4810, 0.5822, 0.7230, 8.0723060e-01, 8.526136257686387498e-01, 0.874616529465895]
    auc_unique = [0.7307, 0.6050, 0.76426, 6.7548488e-01, 7.195311596716382763e-01, 0.7683388252977222]
    aupr_unique = [0.2365, 0.1924, 0.2820, 3.8079816e-01, 4.315107643611781341e-01, 0.4514638964184672]
    rob_exp1 = ["rm DDI", "rm PPI", "rm both", 'All networks']
    rob_exp1_auc = [0.9534857967073433, 0.9490886135877215, 0.9492768825606219, 0.96403536899248]
    rob_exp1_aupr = [0.8677673684253403, 0.8491222010636097, 0.8442767572185182, 0.8743130229290721]
    rob_exp2 = ["rm drug_sim", "rm protein_sim", "rm both", 'All networks']
    rob_exp2_auc = [0.9531893823248611, 0.9539570739781231, 0.9539762112937021, 0.96403536899248]
    rob_exp2_aupr = [0.872520306148556, 0.8724976528127358, 0.8702726047853293, 0.8743130229290721]
    rob_exp3 = ["rm sideeffect", "rm disease", "rm both", 'All networks']
    rob_exp3_auc = [0.9531893823248611, 0.9463327278773364, 0.9434917844684427, 0.96403536899248]
    rob_exp3_aupr = [0.8721869988526759, 0.8414830970065355, 0.8367324812273214, 0.8743130229290721]
    x = np.arange(0, 2, step=.5)
    y = x ** 2
    norm = plt.Normalize(y.min(), y.max())
    norm_y = norm(y)
    map_vir = cm.get_cmap(name='viridis')
    color1 = map_vir(norm_y)

    under_name = rob_exp3
    name = testi
    file_name = 'testi'
    l1, h1, l2, h2 = 0.9, 1.0, 0.8, 0.9
    auc = rob_exp3_auc
    aupr = rob_exp3_aupr
    double(under_name, auc, aupr, color1, name, file_name, l1, h1, l2, h2)
