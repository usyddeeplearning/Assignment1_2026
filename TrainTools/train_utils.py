"""
train_utils.py — Low-level training utilities used by train().
"""

import os

import numpy as np
import torch
from tqdm import tqdm


def train_single_epoch(
    model,
    optimizer,
    scheduler,
    data_iter,
    steps,
    grad_clip,
    loss_fn,
    device,
    global_step: int = 0,
) -> float:
    """
    Run one block of `steps` training iterations consuming from `data_iter`.
    Returns the mean loss over this block.
    """
    model.train()
    loss_list = []

    for _ in tqdm(range(steps), total=steps):
        Cwid, Ccid, Qwid, Qcid, y1, y2, _ = next(data_iter)
        Cwid, Ccid = Cwid.to(device), Ccid.to(device)
        Qwid, Qcid = Qwid.to(device), Qcid.to(device)
        y1, y2 = y1.to(device), y2.to(device)

        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        loss = loss_fn(p1, p2, y1, y2)
        loss_list.append(float(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

    mean_loss = float(np.mean(loss_list))
    print(f"STEP {global_step + steps:8d}  loss {mean_loss:8f}\n")
    return mean_loss


def save_checkpoint(
    save_dir,
    ckpt_name,
    model,
    optimizer,
    scheduler,
    step,
    best_f1,
    best_em,
    config,
    dev_f1=None,
    dev_em=None,
):
    """Save the selected checkpoint state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
        "best_f1": best_f1,
        "best_em": best_em,
        "config": config,
        "selection_metric": "dev_f1_then_dev_em",
    }
    if dev_f1 is not None:
        payload["dev_f1"] = dev_f1
    if dev_em is not None:
        payload["dev_em"] = dev_em
    torch.save(payload, os.path.join(save_dir, ckpt_name))
