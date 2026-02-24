import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)],
            dtype=torch.float32
        ).unsqueeze(1)  # [C, 1]
        phases = torch.tensor(
            [0.0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)],
            dtype=torch.float32
        ).unsqueeze(1)
        pos = torch.arange(length, dtype=torch.float32).repeat(d_model, 1)
        pe = torch.sin(pos * freqs + phases)  # [C, L]
        self.register_buffer("pos_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        return x + self.pos_encoding[:, :length]


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dim: int = 1, bias: bool = True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_ch, in_ch, k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_ch, out_ch, 1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_ch, in_ch, k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=bias)
        else:
            raise ValueError("dim must be 1 or 2")

        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        if self.depthwise_conv.bias is not None:
            nn.init.constant_(self.depthwise_conv.bias, 0.0)

        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        if self.pointwise_conv.bias is not None:
            nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> [B, L, C]
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1.0 - gate) * x
        return x.transpose(1, 2)


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
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L], mask: [B, L] True=PAD
        batch_size, channels, length = x.size()
        x = x.transpose(1, 2)  # [B, L, C]

        q = self.q_linear(x).view(batch_size, length, self.num_heads, self.d_k)
        k = self.k_linear(x).view(batch_size, length, self.num_heads, self.d_k)
        v = self.v_linear(x).view(batch_size, length, self.num_heads, self.d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, length, self.d_k)

        if mask.dtype != torch.bool:
            mask = mask.bool()
        attn_mask = mask.unsqueeze(1).expand(-1, length, -1).repeat(self.num_heads, 1, 1)  # [B*h, L, L]

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = mask_logits(attn, attn_mask)
        attn = F.softmax(attn, dim=2)
        attn = self.drop(attn)

        out = torch.bmm(attn, v)  # [B*h, L, d_k]
        out = out.view(self.num_heads, batch_size, length, self.d_k)
        out = out.permute(1, 2, 0, 3).contiguous().view(batch_size, length, self.d_model)
        out = self.fc(out)
        out = self.drop(out)
        return out.transpose(1, 2)  # [B, C, L]


class Embedding(nn.Module):
    def __init__(self, d_word: int, d_char: int, dropout: float, dropout_char: float):
        super().__init__()
        self.dropout = dropout
        self.dropout_char = dropout_char
        self.conv2d = DepthwiseSeparableConv(d_char, d_char, 5, dim=2)
        self.high = Highway(2, d_word + d_char)

    def forward(self, ch_emb: torch.Tensor, wd_emb: torch.Tensor) -> torch.Tensor:
        # ch_emb: [B, L, char_len, d_char]
        # wd_emb: [B, L, d_word]
        ch_emb = ch_emb.permute(0, 3, 1, 2)  # [B, d_char, L, char_len]
        ch_emb = F.dropout(ch_emb, p=self.dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)  # [B, d_char, L]

        wd_emb = F.dropout(wd_emb, p=self.dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)  # [B, d_word, L]

        emb = torch.cat([ch_emb, wd_emb], dim=1)  # [B, d_char+d_word, L]
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, conv_num: int, k: int, length: int):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, k) for _ in range(conv_num)])
        self.self_att = MultiHeadAttention(d_model, num_heads, dropout)
        self.fc = nn.Linear(d_model, d_model, bias=True)
        self.pos = PosEncoder(d_model, length)

        # Keep original repo behavior: LN over [C, L], so fixed length is required.
        self.normb = nn.LayerNorm([d_model, length])
        self.norms = nn.ModuleList([nn.LayerNorm([d_model, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([d_model, length])
        self.L = conv_num

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.pos(x)
        res = out
        out = self.normb(out)

        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)

        out = self.self_att(out, mask)
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)

        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out
