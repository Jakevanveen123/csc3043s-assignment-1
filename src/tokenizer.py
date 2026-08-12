import json
import time
from collections import Counter, defaultdict

import numpy as np
import regex as re    

END_OF_TEXT = "<|endoftext|>"


PRETOKEN_RE = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def pretokenize(text):

    pretokens = []

    for match in PRETOKEN_RE.finditer(text):
        pretokens.append(match.group(0).encode("utf-8"))

    return pretokens

def count_pre_tokens(text,special_tokens=(END_OF_TEXT,)):

    counts = Counter()
    split_pattern = "|".join(re.escape(s) for s in special_tokens)
    for document in re.split(split_pattern, text):
        counts.update(pretokenize(document))
    return counts

def merge_word(symbols, pair, new_id):
    """
    Replace every occurrence of `pair` in `symbols` with the merged symbol.
    E.g. merge_word([110, 101, 115, 116], (115, 116), 300) -> [110, 101, 300]
    params:
        symbols: list of token ids representing one pre-token
        pair:    (left_id, right_id) to merge
        new_id:  id of the merged symbol
    returns:
        new list of token ids
    """
    merged = []
    i = 0
    while i < len(symbols):
        # If the pair starts here, append the merged symbol and skip two.
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        # Otherwise keep the current symbol and advance by one.
        else:
            merged.append(symbols[i])
            i += 1
    return merged
