"""
train.py — Public training entry point for QANet.

Usage:
    from train import train
    results = train(optimizer_name="adam", num_steps=1000, seed=42)
"""

import argparse
import os

import ujson as json
import torch

from Data import SQuADDataset, load_train_dev_eval, load_word_char_mats, sanity_check_cache
from Models import QANet
from Losses import losses
from Optimizers import optimizers, schedulers
from Tools import EMA, run_eval, set_seed
from train_utils import train_single_epoch, save_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    # ── Data paths ────────────────────────────────────────────────────────────
    train_npz:          str   = "_data/train.npz",
    dev_npz:            str   = "_data/dev.npz",
    word_emb_json:      str   = "_data/word_emb.json",
    char_emb_json:      str   = "_data/char_emb.json",
    train_eval_json:    str   = "_data/train_eval.json",
    dev_eval_json:      str   = "_data/dev_eval.json",
    save_dir:           str   = "_model",
    log_dir:            str   = "_log",
    ckpt_name:          str   = "model.pt",

    # ── Training loop ─────────────────────────────────────────────────────────
    batch_size:         int   = 8,
    num_steps:          int   = 60000,
    checkpoint:         int   = 200,
    val_num_batches:    int   = 150,
    test_num_batches:   int   = 150,
    seed:               int   = 42,
    grad_clip:          float = 5.0,
    early_stop:         int   = 10,

    # ── DL technique selection (string registry keys) ─────────────────────────
    optimizer_name:     str   = "adam",
    scheduler_name:     str   = "warmup_lambda",
    loss_name:          str   = "qa_nll",

    # ── Optimizer hyperparameters ─────────────────────────────────────────────
    learning_rate:      float = 1e-3,
    lr_warm_up_num:     int   = 1000,
    beta1:              float = 0.8,
    beta2:              float = 0.999,
    eps:                float = 1e-7,
    weight_decay:       float = 3e-7,
    momentum:           float = 0.9,    # used by SGD

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_decay:          float = 0.9999,
    use_ema_eval:       bool  = True,

    # ── Model architecture ────────────────────────────────────────────────────
    para_limit:         int   = 400,
    ques_limit:         int   = 50,
    char_limit:         int   = 16,
    d_model:            int   = 96,
    num_heads:          int   = 8,
    glove_dim:          int   = 300,
    char_dim:           int   = 64,
    dropout:            float = 0.1,
    dropout_char:       float = 0.05,
    pretrained_char:    bool  = False,

    # ── Stubs for bug-injection phase ─────────────────────────────────────────
    use_batch_norm:     bool  = False,
    activation:         str   = "relu",
) -> dict:
    """
    Train QANet on SQuAD v1.1.

    Returns
    -------
    dict with keys:
        best_f1   : float       — best dev F1 across all checkpoints
        best_em   : float       — best dev EM across all checkpoints
        history   : list[dict]  — per-checkpoint metrics
            keys: step, train_loss, train_f1, train_em,
                  dev_loss, dev_f1, dev_em, lr
        ckpt_path : str         — absolute path to the saved checkpoint
        config    : dict        — full resolved configuration
    """
    set_seed(seed)

    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Internal namespace required by QANet.__init__ and data utilities
    args = argparse.Namespace(**{k: v for k, v in locals().items()})

    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    sanity_check_cache(args)
    word_mat, char_mat   = load_word_char_mats(args)
    model                = QANet(word_mat, char_mat, args).to(DEVICE)
    train_eval, dev_eval = load_train_dev_eval(args)

    # seed must be set before SQuADDataset (shuffles indices in __init__)
    train_dataset = SQuADDataset(train_npz, num_steps,        batch_size)
    dev_dataset   = SQuADDataset(dev_npz,   test_num_batches, batch_size)

    # Validate and select DL components from registries
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available: {list(optimizers.keys())}")
    if scheduler_name not in schedulers:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'. Available: {list(schedulers.keys())}")
    if loss_name not in losses:
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(losses.keys())}")

    params    = (p for p in model.parameters() if p.requires_grad)
    optimizer = optimizers[optimizer_name](params, args)
    scheduler = schedulers[scheduler_name](optimizer, args)
    loss_fn   = losses[loss_name]

    ema = EMA(ema_decay) if ema_decay > 0 else None
    if ema is not None:
        ema.register(model)

    best_f1  = 0.0
    best_em  = 0.0
    patience = 0
    history  = []

    for step0 in range(0, num_steps, checkpoint):
        steps_this_block = min(checkpoint, num_steps - step0)

        train_loss = train_single_epoch(
            model, optimizer, scheduler, ema, train_dataset,
            step0, steps_this_block, grad_clip, loss_fn, DEVICE,
        )

        tr_metrics, _ = run_eval(
            model, train_dataset, train_eval,
            num_batches=val_num_batches, use_random_batches=True,
            ema=ema, use_ema=use_ema_eval, device=DEVICE, loss_fn=loss_fn,
        )
        print("VALID(train) loss {loss:8f}  F1 {f1:8f}  EM {exact_match:8f}\n".format(**tr_metrics))

        dv_metrics, ans = run_eval(
            model, dev_dataset, dev_eval,
            num_batches=test_num_batches, use_random_batches=False,
            ema=ema, use_ema=use_ema_eval, device=DEVICE, loss_fn=loss_fn,
        )
        print("TEST        loss {loss:8f}  F1 {f1:8f}  EM {exact_match:8f}\n".format(**dv_metrics))

        current_lr = scheduler.get_last_lr()
        print("Learning rate:", current_lr)

        history.append({
            "step":       step0 + steps_this_block,
            "train_loss": train_loss,
            "train_f1":   tr_metrics["f1"],
            "train_em":   tr_metrics["exact_match"],
            "dev_loss":   dv_metrics["loss"],
            "dev_f1":     dv_metrics["f1"],
            "dev_em":     dv_metrics["exact_match"],
            "lr":         current_lr[0] if current_lr else None,
        })

        dev_f1 = dv_metrics["f1"]
        dev_em = dv_metrics["exact_match"]

        if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > early_stop:
                print("Early stopping triggered.")
                break
        else:
            patience = 0
            best_f1  = max(best_f1, dev_f1)
            best_em  = max(best_em, dev_em)

        save_checkpoint(
            save_dir, ckpt_name, model, optimizer, scheduler,
            ema, step0 + steps_this_block, best_f1, best_em, vars(args),
        )

        with open(os.path.join(log_dir, "answers.json"), "w") as f:
            json.dump(ans, f)

    print(f"Training finished.  Best F1: {best_f1:.4f}  Best EM: {best_em:.4f}")

    return {
        "best_f1":   best_f1,
        "best_em":   best_em,
        "history":   history,
        "ckpt_path": os.path.abspath(os.path.join(save_dir, ckpt_name)),
        "config":    vars(args),
    }
