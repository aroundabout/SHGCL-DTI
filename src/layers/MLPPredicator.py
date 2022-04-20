import torch.nn as nn
import torch as th

from tools.tools import compute_score


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


class MLPPredicatorV2(nn.Module):
    def __init__(self, n_i, n_o):
        super(MLPPredicatorV2, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_i, int(n_i / 2)),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(int(n_i / 2), int(n_i / 4)),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(int(n_i / 4), n_o),
        )

    def forward(self, dti, h):
        pre = self.linear(h).reshape(-1, 1).to('cuda:3')
        target = th.tensor(dti[:, 2], dtype=th.float).reshape(-1, 1).to('cuda:3')
        return compute_score(pre, target)
