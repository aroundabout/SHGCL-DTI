import torch.nn as nn
import torch as th

from tools.args import parse_argsCO
from tools.tools import compute_score

args = parse_argsCO()
device = args.device

class MLPPredicator(nn.Module):
    def __init__(self, n_i, n_o):
        super(MLPPredicator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_i, int(n_i / 2)),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(int(n_i / 2), int(n_i / 4)),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(int(n_i / 4), n_o),
        )

    def forward(self, h):
        out = self.linear(h)
        return out