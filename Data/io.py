import numpy as np
import ujson as json


def load_word_char_mats(args):
    with open(args.word_emb_json, "r") as f:
        word_mat = np.array(json.load(f), dtype=np.float32)
    with open(args.char_emb_json, "r") as f:
        char_mat = np.array(json.load(f), dtype=np.float32)
    return word_mat, char_mat


def load_train_dev_eval(args):
    with open(args.train_eval_json, "r") as f:
        train_eval = json.load(f)
    with open(args.dev_eval_json, "r") as f:
        dev_eval = json.load(f)
    return train_eval, dev_eval


def load_dev_eval(args):
    with open(args.dev_eval_json, "r") as f:
        dev_eval = json.load(f)
    return dev_eval
