"""
evaluate.py — Evaluation entry point for QANet / Assignment 1.

Usage (from Assignment1.ipynb):
    from evaluate import evaluate
    metrics = evaluate()
    metrics = evaluate(save_dir="_model", ckpt_name="model.pt")

Returns
-------
dict with keys: f1, exact_match, loss
"""

import argparse
import os

import torch
import ujson as json

from Data import SQuADDataset, load_dev_eval, load_word_char_mats
from Losses import losses
from Models import QANet
from Tools import EMA, run_eval


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    # ── Data paths ────────────────────────────────────────────────────────────
    dev_npz:        str   = "_data/dev.npz",
    word_emb_json:  str   = "_data/word_emb.json",
    char_emb_json:  str   = "_data/char_emb.json",
    dev_eval_json:  str   = "_data/dev_eval.json",
    save_dir:       str   = "_model",
    log_dir:        str   = "_log",
    ckpt_name:      str   = "model.pt",

    # ── Eval settings ─────────────────────────────────────────────────────────
    batch_size:         int   = 8,
    test_num_batches:   int   = -1,       # -1 = full dev set
    use_ema_eval:       bool  = True,
    loss_name:          str   = "qa_nll",

    # ── Model architecture (must match the checkpoint) ────────────────────────
    para_limit:     int   = 400,
    ques_limit:     int   = 50,
    char_limit:     int   = 16,
    d_model:        int   = 96,
    num_heads:      int   = 8,
    glove_dim:      int   = 300,
    char_dim:       int   = 64,
    dropout:        float = 0.1,
    dropout_char:   float = 0.05,
    pretrained_char: bool = False,
) -> dict:
    """Evaluate a saved QANet checkpoint on the SQuAD v1.1 dev set.

    Parameters
    ----------
    dev_npz:
        Path to the preprocessed dev record file.
    word_emb_json, char_emb_json:
        Paths to the embedding matrices produced by ``preprocess()``.
    dev_eval_json:
        Path to the dev evaluation metadata (contexts, gold answers).
    save_dir:
        Directory containing the checkpoint file.
    log_dir:
        Directory where ``answers.json`` will be written.
    ckpt_name:
        Filename of the checkpoint inside ``save_dir``.
    batch_size:
        Number of examples per batch.
    test_num_batches:
        Number of batches to evaluate (-1 = entire dev set).
    use_ema_eval:
        Use EMA shadow weights if available in the checkpoint.
    loss_name:
        Loss function key from the registry (default ``"qa_nll"``).
    para_limit, ques_limit, char_limit, d_model, num_heads,
    glove_dim, char_dim, dropout, dropout_char, pretrained_char:
        Model architecture parameters — must match the values used during
        training.

    Returns
    -------
    dict
        ``{"f1": float, "exact_match": float, "loss": float}``
    """
    os.makedirs(log_dir, exist_ok=True)

    if loss_name not in losses:
        raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(losses.keys())}")

    # Build a lightweight namespace so existing helpers can consume it
    args = argparse.Namespace(
        dev_npz=dev_npz,
        word_emb_json=word_emb_json,
        char_emb_json=char_emb_json,
        dev_eval_json=dev_eval_json,
        para_limit=para_limit,
        ques_limit=ques_limit,
        char_limit=char_limit,
        d_model=d_model,
        num_heads=num_heads,
        glove_dim=glove_dim,
        char_dim=char_dim,
        dropout=dropout,
        dropout_char=dropout_char,
        pretrained_char=pretrained_char,
    )

    word_mat, char_mat = load_word_char_mats(args)
    model = QANet(word_mat, char_mat, args).to(DEVICE)

    dev_eval = load_dev_eval(args)
    dev_dataset = SQuADDataset(dev_npz, test_num_batches, batch_size)

    ckpt_path = os.path.join(save_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    ema = None
    if use_ema_eval and "ema_shadow" in ckpt:
        ema = EMA(ckpt.get("ema_decay", 0.0))
        ema.shadow = {k: v.to(DEVICE) for k, v in ckpt["ema_shadow"].items()}

    metrics, ans = run_eval(
        model, dev_dataset, dev_eval,
        num_batches=test_num_batches,
        use_random_batches=False,
        ema=ema,
        use_ema=use_ema_eval,
        device=DEVICE,
        loss_fn=losses[loss_name],
    )

    with open(os.path.join(log_dir, "answers.json"), "w") as f:
        json.dump(ans, f)

    print("TEST  loss {loss:.6f}  F1 {f1:.6f}  EM {exact_match:.6f}".format(**metrics))
    return {"f1": metrics["f1"], "exact_match": metrics["exact_match"], "loss": metrics["loss"]}
