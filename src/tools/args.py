import argparse


def parse_argsCO():
    parser = argparse.ArgumentParser(description='CoDTI')
    parser.add_argument("--device", type=str, default='cuda:3')
    parser.add_argument("--random_state", type=int, default=7,
                        help="")
    parser.add_argument("--epochs", type=int, default=3000,
                        help="number of training epochs")
    parser.add_argument("--rounds", type=int, default=1,
                        help="number of training rounds")
    # 公用
    parser.add_argument("--in_dim", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--hid_dim", type=int, default=512,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--l2', type=float, default=0.0,
                        help="l2 parameter")
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience.')
    parser.add_argument("--feat_drop", type=float, default=0.5,
                        help="feat dropout")
    parser.add_argument("--attn_drop", type=float, default=0.3,
                        help="attn drop")
    # HECO
    parser.add_argument("--tau", type=float, default=0.5,
                        help='')
    parser.add_argument("--lam", type=float, default=0.5,
                        help='')
    # GAT
    parser.add_argument("--multi_head", type=int, default=5,
                        help='')
    return parser.parse_args()
