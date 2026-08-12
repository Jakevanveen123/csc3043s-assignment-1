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

