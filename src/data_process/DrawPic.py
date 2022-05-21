import matplotlib.pyplot as plt


def draw_pic(exp_list, aupr_list, auc_list, c1, c2):
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 输出图像的标题可以为中文正常输出
    plt.rcParams["axes.unicode_minus"] = False  # 可以正常输出图线里的负号
    plt.rcParams['figure.figsize'] = (16.0, 6.0)

    x = list(range(len(auc_list)))
    plt.autoscale(enable=True, axis='y', tight=None)
    total_width, n = 2, 4
    width = total_width / n
    plt.bar(x, auc_list, width=width, label="AUROC", tick_label=exp_list, color=c1, ec='black')
    for a, b in zip(x, auc_list):
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, aupr_list, width=width, label="AUPR", tick_label=exp_list, color=c2, ec='black')
    # plt.xlabel("不同评分目标类型", fontsize=10)
    # plt.ylabel("不同模型的准确率大小", fontsize=10)
    plt.title("不同评分结果的机器学习算法模型精度表现", fontsize=15)
    for a, b in zip(x, aupr_list):  # 柱子上的数字显示
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    plt.legend(fontsize=8)
    plt.show()


if __name__ == '__main__':
    color1 = 'seagreen'
    color2 = 'khaki'
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

    auc_unique = [0.7307, 0.6050, 0.7686, 6.7548488e-01, 7.195311596716382763e-01, 0.7683388252977222]
    aupr_unique = [0.2365, 0.1924, 0.2820, 3.8079816e-01, 4.315107643611781341e-01, 0.4514638964184672]

    rob_exp1 = ["remove DDI", "remove PPI", "remove DDI and PPI", 'All networks']
    rob_exp1_auc = [0.9534857967073433, 0.9490886135877215, 0.9470886135877215, 0.96403536899248]
    rob_exp1_aupr = [0.8677673684253403, 0.8491222010636097, 0.8451222010636097, 0.8743130229290721]

    rob_exp2 = ["remove sideeffect", "remove disease", "remove both", 'All networks']
    rob_exp2_auc = [0.9531893823248611, 0.9463327278773364, 0.9434917844684427, 0.96403536899248]
    rob_exp2_aupr = [0.8721869988526759, 0.8414830970065355, 0.8367324812273214, 0.8743130229290721]

    rob_exp3 = ["remove drug_similarity", "remove protein_similarity", "remove both similarity", 'All networks']
    rob_exp3_auc = [0.9531893823248611, 0.9539570739781231, 0.9539762112937021, 0.96403536899248]
    rob_exp3_aupr = [0.872520306148556, 0.8724976528127358, 0.8702726047853293, 0.8743130229290721]

    draw_pic(com_exp, aupr_benchmark, auc_benchmark, color1, color2)
