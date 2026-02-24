import torch
import torch.nn as nn


class EMA:
    def __init__(self, decay: float):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

    def register(self, model: nn.Module):
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * p.detach()
            self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def apply(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.copy_(self.backup[name])
        self.backup = {}
