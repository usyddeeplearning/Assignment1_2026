"""
preproc.py — Preprocessing entry point for QANet / Assignment 1.

Usage (from Assignment1.ipynb):
    from Tools.preproc import preprocess
    preprocess()                        # all defaults (_data/ layout)
    preprocess(
        train_file="_data/squad/train-v1.1.json",
        dev_file="_data/squad/dev-v1.1.json",
        glove_word_file="_data/glove/glove.840B.300d.txt",
        target_dir="_data",
        para_limit=400,
        ques_limit=50,
    )

Outputs written to target_dir:
    train.npz, dev.npz
    word_emb.json, char_emb.json
    train_eval.json, dev_eval.json
    word2idx.json, char2idx.json
    dev_meta.json
"""

import os
import re
from collections import Counter

import numpy as np
import ujson as json
from tqdm import tqdm


"""
The preprocessing logic is mostly adapted from:
https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
"""


_TOKENIZER = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def word_tokenize(sent: str):
    return _TOKENIZER.findall(sent)


def convert_idx(text: str, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            raise ValueError(f"Token {token!r} cannot be found in context")
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename: str, data_type: str, word_counter: Counter, char_counter: Counter):
    """Parse a SQuAD JSON file and collect examples + word/char counts."""
    print(f"Generating {data_type} examples…")
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s, answer_texts = [], [], []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer["answer_start"]
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = [
                            idx for idx, span in enumerate(spans)
                            if not (answer_end <= span[0] or answer_start >= span[1])
                        ]
                        y1s.append(answer_span[0])
                        y2s.append(answer_span[-1])
                    examples.append({
                        "context_tokens": context_tokens,
                        "context_chars": context_chars,
                        "ques_tokens": ques_tokens,
                        "ques_chars": ques_chars,
                        "y1s": y1s,
                        "y2s": y2s,
                        "id": total,
                    })
                    eval_examples[str(total)] = {
                        "context": context,
                        "spans": spans,
                        "answers": answer_texts,
                        "uuid": qa["id"],
                    }
    print(f"  {len(examples)} questions in total")
    return examples, eval_examples


def get_embedding(counter: Counter, data_type: str, limit: int = -1,
                  emb_file: str = None, vec_size: int = None):
    """Build an embedding matrix and token→index mapping."""
    print(f"Generating {data_type} embedding…")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]

    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                array = line.split()
                word = "".join(array[:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(f"  {len(embedding_dict)} / {len(filtered_elements)} tokens have "
              f"a corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        print(f"  {len(filtered_elements)} tokens have a corresponding embedding vector")

    null_tok, oov_tok = "--NULL--", "--OOV--"
    token2idx = {tok: idx for idx, tok in enumerate(embedding_dict.keys(), 2)}
    token2idx[null_tok] = 0
    token2idx[oov_tok] = 1
    embedding_dict[null_tok] = [0.0] * vec_size
    embedding_dict[oov_tok] = [0.0] * vec_size
    idx2emb = {idx: embedding_dict[tok] for tok, idx in token2idx.items()}
    emb_mat = [idx2emb[i] for i in range(len(idx2emb))]
    return emb_mat, token2idx


def build_features(examples, data_type: str, out_file: str,
                   word2idx: dict, char2idx: dict,
                   para_limit: int, ques_limit: int,
                   ans_limit: int, char_limit: int):
    """Vectorise examples into padded index arrays and save as .npz."""

    def filter_func(ex):
        return (
            len(ex["context_tokens"]) > para_limit
            or len(ex["ques_tokens"]) > ques_limit
            or (ex["y2s"][0] - ex["y1s"][0]) > ans_limit
        )

    def get_word(word):
        for variant in (word, word.lower(), word.capitalize(), word.upper()):
            if variant in word2idx:
                return word2idx[variant]
        return 1  # OOV

    def get_char(char):
        return char2idx.get(char, 1)  # OOV

    print(f"Processing {data_type} examples…")
    total = total_all = 0
    context_idxs, context_char_idxs = [], []
    ques_idxs, ques_char_idxs = [], []
    y1s, y2s, ids = [], [], []

    for example in tqdm(examples):
        total_all += 1
        if filter_func(example):
            continue
        total += 1

        ctx_idx = np.zeros([para_limit], dtype=np.int32)
        ctx_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        q_idx = np.zeros([ques_limit], dtype=np.int32)
        q_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            ctx_idx[i] = get_word(token)
        for i, token in enumerate(example["ques_tokens"]):
            q_idx[i] = get_word(token)
        for i, chars in enumerate(example["context_chars"]):
            for j, ch in enumerate(chars[:char_limit]):
                ctx_char_idx[i, j] = get_char(ch)
        for i, chars in enumerate(example["ques_chars"]):
            for j, ch in enumerate(chars[:char_limit]):
                q_char_idx[i, j] = get_char(ch)

        context_idxs.append(ctx_idx)
        context_char_idxs.append(ctx_char_idx)
        ques_idxs.append(q_idx)
        ques_char_idxs.append(q_char_idx)
        y1s.append(example["y1s"][-1])
        y2s.append(example["y2s"][-1])
        ids.append(example["id"])

    ensure_parent(out_file)
    np.savez(
        out_file,
        context_idxs=np.array(context_idxs),
        context_char_idxs=np.array(context_char_idxs),
        ques_idxs=np.array(ques_idxs),
        ques_char_idxs=np.array(ques_char_idxs),
        y1s=np.array(y1s),
        y2s=np.array(y2s),
        ids=np.array(ids),
    )
    print(f"  Built {total} / {total_all} instances")
    return {"total": total}


def save_json(filename: str, obj, message: str = None) -> None:
    if message:
        print(f"Saving {message}…")
    ensure_parent(filename)
    with open(filename, "w") as fh:
        json.dump(obj, fh)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(
    # --- Raw data inputs ---
    train_file: str = os.path.join("_data", "squad", "train-v1.1.json"),
    dev_file: str = os.path.join("_data", "squad", "dev-v1.1.json"),
    glove_word_file: str = os.path.join("_data", "glove", "glove.840B.300d.txt"),
    glove_char_file: str = None,
    fasttext_file: str = None,
    # --- Output directories ---
    target_dir: str = "_data",
    save_dir: str = "_model",
    log_dir: str = "_log",
    # --- Shape / limit parameters ---
    glove_dim: int = 300,
    char_dim: int = 64,
    para_limit: int = 400,
    ques_limit: int = 50,
    ans_limit: int = 30,
    char_limit: int = 16,
    word_count_limit: int = -1,
    char_count_limit: int = -1,
    # --- Embedding options ---
    pretrained_char: bool = False,
    fasttext: bool = False,
) -> dict:
    """Preprocess SQuAD v1.1 for QANet training.

    Reads raw SQuAD JSON files and GloVe vectors, then writes padded index
    tensors (.npz) and auxiliary JSON files to *target_dir*.

    Parameters
    ----------
    train_file, dev_file:
        Paths to SQuAD v1.1 train/dev JSON files.
    glove_word_file:
        Path to the GloVe 840B 300d ``.txt`` file.
    glove_char_file:
        Path to a character-level GloVe file (only used when
        ``pretrained_char=True``).
    fasttext_file:
        Path to a fastText ``.vec`` file (only used when ``fasttext=True``).
    target_dir:
        Root directory for all output files (default ``"_data"``).
    save_dir:
        Directory for model checkpoints; created if absent (default
        ``"_model"``).
    log_dir:
        Directory for log / answer files; created if absent (default
        ``"_log"``).
    glove_dim:
        Dimension of GloVe word vectors (default 300).
    char_dim:
        Dimension of randomly-initialised character vectors (default 64).
    para_limit:
        Maximum number of tokens in a context paragraph (default 400).
    ques_limit:
        Maximum number of tokens in a question (default 50).
    ans_limit:
        Maximum answer span length in tokens (default 30).
    char_limit:
        Maximum characters per token (default 16).
    word_count_limit:
        Minimum word frequency to keep (-1 = keep all, default -1).
    char_count_limit:
        Minimum char frequency to keep (-1 = keep all, default -1).
    pretrained_char:
        Use ``glove_char_file`` for character embeddings (default False).
    fasttext:
        Use ``fasttext_file`` instead of GloVe for word embeddings
        (default False).

    Returns
    -------
    dict
        Paths of all output files::

            {
                "train_record_file": "...",
                "dev_record_file":   "...",
                "word_emb_file":     "...",
                "char_emb_file":     "...",
                "train_eval_file":   "...",
                "dev_eval_file":     "...",
                "word2idx_file":     "...",
                "char2idx_file":     "...",
                "dev_meta_file":     "...",
            }
    """
    # Create output directories
    for d in (target_dir, save_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    # Derive output file paths
    out = {
        "train_record_file": os.path.join(target_dir, "train.npz"),
        "dev_record_file":   os.path.join(target_dir, "dev.npz"),
        "word_emb_file":     os.path.join(target_dir, "word_emb.json"),
        "char_emb_file":     os.path.join(target_dir, "char_emb.json"),
        "train_eval_file":   os.path.join(target_dir, "train_eval.json"),
        "dev_eval_file":     os.path.join(target_dir, "dev_eval.json"),
        "word2idx_file":     os.path.join(target_dir, "word2idx.json"),
        "char2idx_file":     os.path.join(target_dir, "char2idx.json"),
        "dev_meta_file":     os.path.join(target_dir, "dev_meta.json"),
    }

    # Step 1 — Parse SQuAD files
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(dev_file, "dev", word_counter, char_counter)

    # Step 2 — Build embedding matrices
    word_emb_source = fasttext_file if fasttext else glove_word_file
    char_emb_source = glove_char_file if pretrained_char else None
    char_emb_dim = glove_dim if pretrained_char else char_dim

    word_emb_mat, word2idx = get_embedding(
        word_counter, "word",
        limit=word_count_limit,
        emb_file=word_emb_source,
        vec_size=glove_dim,
    )
    char_emb_mat, char2idx = get_embedding(
        char_counter, "char",
        limit=char_count_limit,
        emb_file=char_emb_source,
        vec_size=char_emb_dim,
    )

    # Step 3 — Vectorise and save .npz records
    build_features(
        train_examples, "train", out["train_record_file"],
        word2idx, char2idx,
        para_limit, ques_limit, ans_limit, char_limit,
    )
    dev_meta = build_features(
        dev_examples, "dev", out["dev_record_file"],
        word2idx, char2idx,
        para_limit, ques_limit, ans_limit, char_limit,
    )

    # Step 4 — Save JSON files
    save_json(out["word_emb_file"],   word_emb_mat,  "word embedding")
    save_json(out["char_emb_file"],   char_emb_mat,  "char embedding")
    save_json(out["train_eval_file"], train_eval,    "train eval")
    save_json(out["dev_eval_file"],   dev_eval,      "dev eval")
    save_json(out["word2idx_file"],   word2idx,      "word dictionary")
    save_json(out["char2idx_file"],   char2idx,      "char dictionary")
    save_json(out["dev_meta_file"],   dev_meta,      "dev meta")

    print("\nPreprocessing complete.")
    print(f"  Outputs → {target_dir}/")
    return out
