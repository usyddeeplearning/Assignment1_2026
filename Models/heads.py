import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import mask_logits


class CQAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        w = torch.empty(d_model * 3)
        lim = 1.0 / d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C: torch.Tensor, Q: torch.Tensor, cmask: torch.Tensor, qmask: torch.Tensor) -> torch.Tensor:
        # C: [B, C, Lc], Q: [B, C, Lq]
        C = C.transpose(1, 2)  # [B, Lc, C]
        Q = Q.transpose(1, 2)  # [B, Lq, C]

        cmask = cmask.unsqueeze(2)  # [B, Lc, 1]
        qmask = qmask.unsqueeze(1)  # [B, 1, Lq]

        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))  # [B, Lc, Lq, C]
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = Ct * Qt
        S = torch.cat([Ct, Qt, CQ], dim=3)  # [B, Lc, Lq, 3C]
        S = torch.matmul(S, self.w)  # [B, Lc, Lq]

        S1 = F.softmax(mask_logits(S, qmask), dim=2)
        S2 = F.softmax(mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)

        out = torch.cat([C, A, C * A, C * B], dim=2)  # [B, Lc, 4C]
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out.transpose(1, 2)  # [B, 4C, Lc]


class Pointer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        w1 = torch.empty(d_model * 2)
        w2 = torch.empty(d_model * 2)
        lim = 3.0 / (2.0 * d_model)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1: torch.Tensor, M2: torch.Tensor, M3: torch.Tensor, mask: torch.Tensor):
        X1 = torch.cat([M1, M2], dim=1)  # [B, 2C, L]
        X2 = torch.cat([M1, M3], dim=1)  # [B, 2C, L]
        Y1 = torch.matmul(self.w1, X1)  # [B, L]
        Y2 = torch.matmul(self.w2, X2)  # [B, L]
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2
