import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CoDTI')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--epochs", type=int, default=5000,
                        help="number of training epochs")
    # 公用
    parser.add_argument("--in_dim", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--hid_dim", type=int, default=2048,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--patience', type=int, default=500,
                        help='Early stopping patience.')
    # feat drop
    parser.add_argument("--feat_drop", type=float, default=0.5,
                        help="feat dropout")
    parser.add_argument("--attn_drop", type=float, default=0.2,
                        help="attn drop")
    parser.add_argument("--multi_head", type=int, default=5,
                        help='')
    # contrast
    parser.add_argument("--tau", type=float, default=0.5,
                        help='')
    parser.add_argument("--lam", type=float, default=0.5,
                        help='')
    parser.add_argument("--cl", type=float, default=0,
                        help='')
    parser.add_argument("--reg_lambda", type=float, default=0.5,
                        help='')
    # 控制实验
    parser.add_argument("--number", type=str, default='ten',
                        help='控制负样本比例 one1:1 ten为1:10 all为 全量')
    parser.add_argument('--task', type=str, default='benchmark',
                        help='实验分为正常数据benchmark,其他是disease,drug,homo_protein_drug,sideeffect,uni等')
    parser.add_argument('--edge_mask', type=str, default='',
                        help='控制实验中是否要mask某些边 drug protein drug,protein disease sideeffect disease,sideeffect'
                             'drugsim proteinsim drugsim,proteinsim')
    return parser.parse_args()
