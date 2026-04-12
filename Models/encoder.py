import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import DepthwiseSeparableConv
from .dropout import Dropout
from .Normalizations import get_norm
from .Activations import get_activation


def mask_logits(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mask: True means masked (PAD) positions.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return target.masked_fill(mask, -1e30)


class PosEncoder(nn.Module):
    """
    Sinusoidal positional encoding as a non-trainable buffer.
    x: [B, C, L]
    """

    def __init__(self, d_model: int, length: int):
        super().__init__()
        freqs = torch.tensor(
            [
                (
                    10000 ** (-i / d_model)
                    if i % 2 == 0
                    else -(10000 ** ((1 - i) / d_model))
                )
                for i in range(d_model)
            ],
            dtype=torch.float32,
        ).unsqueeze(
            1
        )  # [C, 1]
        phases = torch.tensor(
            [0.0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)],
            dtype=torch.float32,
        ).unsqueeze(1)
        pos = torch.arange(length, dtype=torch.float32).repeat(d_model, 1)
        pe = torch.sin(pos * freqs + phases)  # [C, L]
        self.register_buffer("pos_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        return x + self.pos_encoding[:, :length]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.drop = Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L], mask: [B, L] True=PAD
        batch_size, channels, length = x.size()
        x = x.transpose(1, 2)  # [B, L, C]

        q = self.q_linear(x).view(batch_size, length, self.num_heads, self.d_k)
        k = self.k_linear(x).view(batch_size, length, self.num_heads, self.d_k)
        v = self.v_linear(x).view(batch_size, length, self.num_heads, self.d_k)

        q = (
            q.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, length, self.d_k)
        )
        k = (
            k.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, length, self.d_k)
        )
        v = (
            v.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, length, self.d_k)
        )

        if mask.dtype != torch.bool:
            mask = mask.bool()
        attn_mask = (
            mask.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, self.num_heads, length, -1)
            .reshape(batch_size * self.num_heads, length, length)
        )  # [B*h, L, L]

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = mask_logits(attn, attn_mask)
        attn = F.softmax(attn, dim=2)
        attn = self.drop(attn)

        out = torch.bmm(attn, v)  # [B*h, L, d_k]
        out = out.view(batch_size, self.num_heads, length, self.d_k)
        out = (
            out.permute(0, 2, 1, 3).contiguous().view(batch_size, length, self.d_model)
        )
        out = self.fc(out)
        out = self.drop(out)
        return out.transpose(1, 2)  # [B, C, L]


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        conv_num: int,
        k: int,
        length: int,
        init_name: str = "kaiming",
        act_name: str = "relu",
        norm_name: str = "layer_norm",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                DepthwiseSeparableConv(d_model, d_model, k, init_name=init_name)
                for _ in range(conv_num)
            ]
        )
        # Stochastic-depth dropout: p scales linearly with layer depth.
        self.conv_drops = nn.ModuleList(
            [Dropout(dropout * (i + 1) / conv_num) for i in range(conv_num)]
        )
        self.drop = Dropout(dropout)
        self.self_att = MultiHeadAttention(d_model, num_heads, dropout)
        self.fc = nn.Linear(d_model, d_model, bias=True)
        self.pos = PosEncoder(d_model, length)
        self.act = get_activation(act_name)

        # Normalization over [C, L]; fixed length required for layer_norm.
        self.normb = get_norm(norm_name, d_model, length, num_groups=norm_groups)
        self.norms = nn.ModuleList(
            [
                get_norm(norm_name, d_model, length, num_groups=norm_groups)
                for _ in range(conv_num)
            ]
        )
        self.norme = get_norm(norm_name, d_model, length, num_groups=norm_groups)
        self.L = conv_num

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.pos(x)
        res = out
        out = self.normb(out)

        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = self.act(out)
            out = out + res
            out = self.conv_drops[i](out)
            res = out
            out = self.norms[i](out)

        out = self.self_att(out, mask)
        out = res + out
        out = self.drop(out)

        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = out + res
        out = self.drop(out)
        return out
