import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CoDTI')
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--epochs", type=int, default=3000,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048),
    parser.add_argument("--random_state", type=int, default=18,
                        help="")
    # 公用
    parser.add_argument("--in_dim", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--hid_dim", type=int, default=512,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--l2', type=float, default=0.0,
                        help="l2 parameter")
    parser.add_argument('--patience', type=int, default=40,
                        help='Early stopping patience.')
    # feat drop
    parser.add_argument("--feat_drop", type=float, default=0.5,
                        help="feat dropout")
    # sc encoder for gat
    parser.add_argument("--attn_drop", type=float, default=0.3,
                        help="attn drop")
    parser.add_argument("--multi_head", type=int, default=5,
                        help='')
    # contrast
    parser.add_argument("--tau", type=float, default=0.5,
                        help='')
    parser.add_argument("--lam", type=float, default=0.5,
                        help='')

    return parser.parse_args()
