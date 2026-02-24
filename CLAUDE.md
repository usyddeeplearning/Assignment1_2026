# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Background

This repository is the **codebase for Assignment 1** of the Deep Learning course at the University of Sydney (USYD), Semester 1, 2026. The course targets Masters and Honours students in relevant programs.

### Assignment structure

The assignment distributes a complete QANet training and evaluation framework. On top of the working model backbone, the assignment covers standard deep learning techniques taught in the course:

- Optimizers (e.g. Adam)
- Learning rate scheduling (e.g. warm-up)
- Loss computation
- Backpropagation logic
- Weight decay
- Dropout
- Batch Normalization
- Activation functions

**The distributed code is intentionally buggy.** There are two categories of bugs students must find and fix:

| Category | Count | Description |
|----------|-------|-------------|
| **Runtime bugs** | ~10 | Basic code errors that prevent the project from running correctly. Students must locate and fix all of them. |
| **Implementation bugs** | ~20 | Incorrect implementations of DL techniques (e.g. a wrong Adam update rule). Students must fix these to ensure each component is correctly implemented. |

### Submission format

- Submissions are made via **Google Colab** (students share a Google Drive link).
- The grading interface is a fixed notebook: **`Assignment1.ipynb`**.
- All other project files are packaged as importable functions. The notebook drives the whole project using calls such as:

```python
from train import train
train(...)
```

The `train` function (and equivalent entry points for other stages) must accept all necessary hyperparameters as arguments — the notebook fills in and passes the args.

### Development progress

**Phase 1 — Refactor (in progress)**

| Status | Task |
|--------|------|
| ✅ | `train.py` — clean `train()` entry point |
| ✅ | `train_utils.py` — `train_single_epoch`, `save_checkpoint` |
| ✅ | `Tools/utils.py` — `set_seed` |
| ✅ | `Optimizers/optimizer.py` — extended with SGD, AdamW, cosine/step schedulers |
| ✅ | `Losses/loss.py` — extended with `qa_ce_loss` |
| ⬜ | `eval.py` — `evaluate()` entry point |
| ⬜ | `preproc.py` — `preprocess()` entry point |
| ⬜ | `Assignment1.ipynb` — grading notebook |
| ⬜ | Retire / shim `01_preproc.py`, `02_train.py`, `03_eval.py` |

**Phase 2 — Bug injection (pending Phase 1 completion)**
- ~10 runtime bugs
- ~20 DL technique implementation bugs

---

## Pipeline

The project is driven from `Assignment1.ipynb`. Each stage is callable as an imported function:

```python
from preproc import preprocess
preprocess(train_file="data/squad/train-v1.1.json", ...)

from train import train
results = train(optimizer_name="adam", num_steps=60000, seed=42)
# results: {"best_f1", "best_em", "history", "ckpt_path", "config"}

from eval import evaluate
metrics = evaluate(save_dir="model", ckpt_name="model.pt")
# metrics: {"f1", "exact_match", "loss"}
```

Data download (one-time setup):
```bash
bash 00_download.sh   # downloads SQuAD v1.1 + GloVe 840B
```

## Architecture

### Data flow

Raw SQuAD JSON + GloVe `.txt` → `preprocess()` → `data/{train,dev}.npz` (padded index tensors) + `data/{word,char}_emb.json` + `data/{train,dev}_eval.json` → `train()`.

`SQuADDataset` ([Data/squad.py](Data/squad.py)) wraps the `.npz` file and returns pre-shuffled batches by step index (not standard DataLoader iteration). `__getitem__(step_idx)` returns one full batch.

### Model forward pass ([Models/qanet.py](Models/qanet.py))

```
(Cwid, Ccid, Qwid, Qcid)
  → word + char embeddings
  → Embedding (char conv2d → max-pool → concat word → Highway)
  → DepthwiseSeparableConv (project to d_model)
  → EncoderBlock (pos encoding → conv stack → self-attention → FC)  [separate for C and Q]
  → CQAttention (trilinear similarity → softmax → [C, A, C*A, C*B])
  → DepthwiseSeparableConv (4*d_model → d_model)
  → 3x passes through shared 7-block EncoderBlock stack → M1, M2, M3
  → Pointer (linear over [M1,M2] and [M1,M3]) → (log_softmax p1, p2)
```

Key architectural constraint: `EncoderBlock` uses `LayerNorm([d_model, length])` — the sequence lengths (`para_limit`, `ques_limit`) must stay fixed between preprocessing and training.

### Module registry pattern

`Losses`, `Optimizers` expose dict registries selected by string key in `train()`:

| Registry | Keys |
|----------|------|
| `optimizers` | `"adam"`, `"sgd"`, `"adamw"` |
| `schedulers` | `"warmup_lambda"`, `"cosine"`, `"step"` |
| `losses`     | `"qa_nll"`, `"qa_ce"` |

Note: `adam` sets internal `lr=1.0` and relies on the `warmup_lambda` scheduler to deliver the actual learning rate. `sgd` and `adamw` use `learning_rate` directly and should be paired with `cosine` or `step`.

### EMA

[Tools/ema.py](Tools/ema.py) maintains a shadow copy of trainable parameters. During eval, `ema.apply(model)` swaps in the shadow weights; `ema.restore(model)` swaps back. Controlled by `ema_decay` and `use_ema_eval` in `train()`.

## Key hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `d_model` | 96 | Model hidden dimension |
| `num_heads` | 8 | Attention heads (must divide d_model) |
| `glove_dim` | 300 | Word embedding dimension |
| `char_dim` | 64 | Char embedding dimension |
| `para_limit` | 400 | Context length (fixed at preproc time) |
| `ques_limit` | 50 | Question length (fixed at preproc time) |
| `batch_size` | 8 | Training batch size |
| `num_steps` | 60000 | Total training steps |
| `checkpoint` | 200 | Eval + save frequency (steps) |
| `lr_warm_up_num` | 1000 | Warmup steps for `warmup_lambda` scheduler |

## Output files

- `model/model.pt` — checkpoint (model, optimizer, scheduler, EMA shadow, best metrics)
- `model/run_config.json` — full config dict for the run
- `log/answers.json` — predicted answers from the last eval pass

## Dependencies

Install PyTorch separately (not in requirements.txt), then:
```bash
pip install -r requirements.txt
python -m spacy download en
```
