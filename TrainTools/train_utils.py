"""
train_utils.py — Low-level training utilities used by train().
"""

import os

import numpy as np
import torch
from tqdm import tqdm


def train_single_epoch(model, optimizer, scheduler, ema, dataset,
                       start_step, steps, grad_clip, loss_fn, device) -> float:
    """
    Run one block of `steps` training iterations starting from `start_step`.
    Returns the mean loss over this block.
    """
    model.train()
    loss_list = []

    for step in tqdm(range(start_step, start_step + steps), total=steps):
        optimizer.zero_grad(set_to_none=True)

        Cwid, Ccid, Qwid, Qcid, y1, y2, _ = dataset[step]
        Cwid, Ccid = Cwid.to(device), Ccid.to(device)
        Qwid, Qcid = Qwid.to(device), Qcid.to(device)
        y1, y2     = y1.to(device),   y2.to(device)

        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        loss   = loss_fn(p1, p2, y1, y2)
        loss_list.append(float(loss.item()))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

    mean_loss = float(np.mean(loss_list))
    print(f"STEP {start_step + steps:8d}  loss {mean_loss:8f}\n")
    return mean_loss


def save_checkpoint(save_dir, ckpt_name, model, optimizer, scheduler,
                    ema, step, best_f1, best_em, config):
    """Save model, optimizer, scheduler and EMA state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step":            step,
        "best_f1":         best_f1,
        "best_em":         best_em,
        "config":          config,
    }
    if ema is not None:
        payload["ema_decay"]  = ema.decay
        payload["ema_shadow"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
    torch.save(payload, os.path.join(save_dir, ckpt_name))
