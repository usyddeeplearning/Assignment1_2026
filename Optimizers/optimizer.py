from math import log2

import torch.optim as optim


# ── Optimizer factories ──────────────────────────────────────────────────────
#
# NOTE: `adam` sets lr=1.0 because its learning rate is entirely controlled by
# the paired `warmup_lambda` scheduler (which outputs the actual lr values).
# `sgd` and `adamw` use args.learning_rate directly and should be paired with
# `cosine` or `step` schedulers.

def adam(params, args):
    return optim.Adam(
        params=params,
        lr=1.0,
        betas=(args.beta1, args.beta2),
        eps=getattr(args, "eps", 1e-7),
        weight_decay=args.weight_decay,
    )


def sgd(params, args):
    return optim.SGD(
        params=params,
        lr=args.learning_rate,
        momentum=getattr(args, "momentum", 0.9),
        weight_decay=args.weight_decay,
    )


def adamw(params, args):
    return optim.AdamW(
        params=params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=getattr(args, "eps", 1e-7),
        weight_decay=args.weight_decay,
    )


# ── Scheduler factories ──────────────────────────────────────────────────────

def warmup_lambda_scheduler(optimizer, args):
    """Logarithmic warmup to args.learning_rate, then constant.
    Designed to pair with the `adam` optimizer (which starts at lr=1.0)."""
    lr = args.learning_rate
    warm_up = args.lr_warm_up_num
    cr = (lr / log2(warm_up)) if warm_up > 1 else lr
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (cr * log2(step + 1)) if step < warm_up else lr,
    )


def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def none_scheduler(optimizer, args):
    """No-op scheduler: learning rate stays constant throughout training."""
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


# ── Registries ───────────────────────────────────────────────────────────────

optimizers = {
    "adam":  adam,
    "sgd":   sgd,
    "adamw": adamw,
}

schedulers = {
    "warmup_lambda": warmup_lambda_scheduler,
    "cosine":        cosine_scheduler,
    "step":          step_scheduler,
    "none":          none_scheduler,
}
