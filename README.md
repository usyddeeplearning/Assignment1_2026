# QANet — COMP5329 Assignment 1

A PyTorch implementation of [QANet](https://arxiv.org/pdf/1804.09541.pdf) for extractive question answering on SQuAD v1.1.
This is the codebase for **Assignment 1** of the Deep Learning course at the University of Sydney (COMP5329, Semester 1 2026).
The entire pipeline is driven from `assignment1.ipynb`; each stage is exposed as a single importable function.

---

## Project Structure

```
assignment1.ipynb          # Grading notebook (Google Colab)
evaluate.py                # evaluate() — load checkpoint and run dev-set inference

TrainTools/
  train.py                 # train() — full training loop
  train_utils.py           # train_single_epoch(), save_checkpoint()

Tools/
  download.py              # download() — fetch SQuAD + GloVe
  preproc.py               # preprocess() — tokenise and build index tensors
  ema.py                   # EMA — exponential moving average of parameters
  metrics.py               # run_eval(), F1 / EM computation
  utils.py                 # set_seed()

Models/
  qanet.py                 # QANet — top-level model
  layers.py                # Embedding, EncoderBlock, DepthwiseSeparableConv, Highway, ...
  heads.py                 # CQAttention, Pointer

Data/
  squad.py                 # SQuADDataset, sanity_check_cache
  io.py                    # load_word_char_mats(), load_dev_eval(), ...

Losses/
  loss.py                  # qa_nll_loss, qa_ce_loss + registry

Optimizers/
  optimizer.py             # adam/sgd/adamw factories + warmup_lambda/cosine/step/none schedulers
```

---

## Pipeline

### 1 — Download (one-time)

```python
from Tools.download import download
download(data_dir="_data")
```

Downloads SQuAD v1.1 (train + dev JSON) and GloVe 840B 300d into `_data/squad/` and `_data/glove/`.
Files are skipped if they already exist.

### 2 — Preprocess (one-time)

```python
from Tools.preproc import preprocess
preprocess(
    train_file       = "_data/squad/train-v1.1.json",
    dev_file         = "_data/squad/dev-v1.1.json",
    glove_word_file  = "_data/glove/glove.840B.300d.txt",
    target_dir       = "_data",
    para_limit       = 400,
    ques_limit       = 50,
)
```

Outputs written to `_data/`:

| File | Description |
|------|-------------|
| `train.npz`, `dev.npz` | Padded word/char index tensors + span labels |
| `word_emb.json`, `char_emb.json` | Embedding matrices (GloVe + random char) |
| `train_eval.json`, `dev_eval.json` | Gold contexts and answers for scoring |
| `word2idx.json`, `char2idx.json` | Vocabulary mappings |

> `para_limit` and `ques_limit` are baked into the `.npz` files and **must match** the values used during training.

### 3 — Train

```python
from TrainTools.train import train

results = train(
    train_npz       = "_data/train.npz",
    dev_npz         = "_data/dev.npz",
    word_emb_json   = "_data/word_emb.json",
    char_emb_json   = "_data/char_emb.json",
    train_eval_json = "_data/train_eval.json",
    dev_eval_json   = "_data/dev_eval.json",
    save_dir        = "_model",
    log_dir         = "_log",
    num_steps       = 60000,
    batch_size      = 8,
    seed            = 42,
    optimizer_name  = "sgd",
    scheduler_name  = "none",
    loss_name       = "qa_nll",
)
# results: {"best_f1", "best_em", "history", "ckpt_path", "config"}
```

Saves the best checkpoint to `_model/model.pt` and logs predictions to `_log/answers.json`.

### 4 — Evaluate

```python
from evaluate import evaluate

metrics = evaluate(
    dev_npz       = "_data/dev.npz",
    word_emb_json = "_data/word_emb.json",
    char_emb_json = "_data/char_emb.json",
    dev_eval_json = "_data/dev_eval.json",
    save_dir      = "_model",
    log_dir       = "_log",
    ckpt_name     = "model.pt",
)
# metrics: {"f1", "exact_match", "loss"}
```

---

## Model Architecture

```
(Cwid, Ccid, Qwid, Qcid)
  → Embedding          char Conv2d → max-pool → concat word emb → Highway
  → EncoderBlock       pos encoding → depthwise-separable conv stack
                       → multi-head self-attention → FC   [separate for C and Q]
  → CQAttention        trilinear similarity → softmax → [C, A, C⊙A, C⊙B]
  → DepthwiseSeparableConv   4×d_model → d_model
  → 3 passes through shared 7-block EncoderBlock stack → M1, M2, M3
  → Pointer            linear([M1,M2]) → p1,  linear([M1,M3]) → p2
                       log_softmax over context length
```

| Component | File | Role |
|-----------|------|------|
| `Embedding` | `Models/layers.py` | Word + char embeddings, Highway network |
| `DepthwiseSeparableConv` | `Models/layers.py` | Efficient 1-D depthwise + pointwise conv |
| `EncoderBlock` | `Models/layers.py` | Positional encoding → conv stack → self-attention → FC |
| `CQAttention` | `Models/heads.py` | Trilinear context–question attention |
| `Pointer` | `Models/heads.py` | Span start/end prediction |
| `QANet` | `Models/qanet.py` | Assembles all components |

---

## DL Components

The codebase covers the following deep learning techniques (the assignment asks students to locate and fix incorrect implementations):

| Category | Options |
|----------|---------|
| **Optimizers** | `adam`, `sgd`, `adamw` |
| **LR Schedulers** | `warmup_lambda` (log warmup), `cosine`, `step`, `none` |
| **Loss functions** | `qa_nll` (NLL on log-softmax), `qa_ce` (cross-entropy on logits) |
| **Regularisation** | Dropout (`dropout`, `dropout_char`), weight decay |
| **Normalisation** | Layer normalisation (`LayerNorm([d_model, length])`) |
| **Optimisation** | Gradient clipping, EMA of parameters |
| **Activations** | ReLU (default), configurable via `activation` |

Optimizers and schedulers are selected by string key and resolved at runtime from registries in `Optimizers/optimizer.py`.
Losses are resolved from the registry in `Losses/loss.py`.

---

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d_model` | 96 | Hidden dimension |
| `num_heads` | 8 | Attention heads (must divide `d_model`) |
| `glove_dim` | 300 | Word embedding dimension |
| `char_dim` | 64 | Character embedding dimension |
| `para_limit` | 400 | Context max length — fixed at preprocess time |
| `ques_limit` | 50 | Question max length — fixed at preprocess time |
| `batch_size` | 8 | Training batch size |
| `num_steps` | 60000 | Total training steps |
| `checkpoint` | 200 | Eval + save frequency (steps) |
| `learning_rate` | 1e-3 | Base LR (adam uses 1.0 internally; scheduler applies the actual value) |
| `lr_warm_up_num` | 1000 | Warmup steps for `warmup_lambda` |
| `ema_decay` | 0.9999 | EMA decay (0 = disabled) |
| `dropout` | 0.1 | General dropout rate |
| `weight_decay` | 3e-7 | L2 penalty |
| `grad_clip` | 5.0 | Gradient clipping norm |
| `early_stop` | 10 | Stop after N checkpoints with no improvement |

---

## Setup (Google Colab)

1. Clone this repo into your Google Drive.
2. Open `assignment1.ipynb` in Google Colab.
3. Run **Section 0** to mount Drive, install dependencies, and set the project path.
4. Run **Sections 1–4** in order (download → preprocess → train → evaluate).

Install dependencies (PyTorch must be installed separately):

```bash
pip install -r requirements.txt
```

---

## Differences from the Paper

1. The paper does not specify an activation function; ReLU is used here.
2. The `<UNK>` token embedding is not set trainable.
3. The connector between the embedding layer and the embedding encoder deviates slightly from the paper (which is inconsistent on this point) — a depthwise-separable projection is used to bridge the dimension mismatch.
