import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CoDTI')
    parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--epochs", type=int, default=3000,
                        help="number of training epochs")
    # parser.add_argument("--batch_size", type=int, default=2048),
    parser.add_argument("--random_state", type=int, default=18,
                        help="")
    # 公用
    parser.add_argument("--in_dim", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--hid_dim", type=int, default=2048,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--l2', type=float, default=0.0,
                        help="l2 parameter")
    parser.add_argument('--patience', type=int, default=250,
                        help='Early stopping patience.')
    # feat drop
    parser.add_argument("--feat_drop", type=float, default=0.5,
                        help="feat dropout")
    # sc encoder for gat
    parser.add_argument("--attn_drop", type=float, default=0.2,
                        help="attn drop")
    parser.add_argument("--multi_head", type=int, default=5,
                        help='')
    # contrast
    parser.add_argument("--tau", type=float, default=0.5,
                        help='')
    parser.add_argument("--lam", type=float, default=0.5,
                        help='')
    parser.add_argument("--cl", type=float, default=5,
                        help='')
    parser.add_argument("--reg_lambda", type=float, default=1,
                        help='')
    # 控制实验
    parser.add_argument("--number", type=str, default='ten',
                        help='控制负样本比例 ten为1:10 all为 全量')
    parser.add_argument('--task', type=str, default='benchmark',
                        help='实验分为正常数据benchmark,其他是disease,drug,homo_protein_drug,se,uni等')

    return parser.parse_args()
