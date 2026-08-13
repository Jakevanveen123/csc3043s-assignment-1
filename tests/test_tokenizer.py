from collections import Counter
import sys
from pathlib import Path
import random
 
import numpy as np
 
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.tokenizer import (                                   
    END_OF_TEXT,
    BPETokenizer,
    count_pretokens,
    merge_word,
    train_bpe,
)

 
DATA_PATH = "data/test_corpus.txt"
 
VOCAB_SIZE = 256 + 1 + 60
 
 
def load_text():
    with open(DATA_PATH, encoding="utf-8") as f:
        return f.read()
 
 
def get_tokenizer():
    vocab, merges = train_bpe(DATA_PATH, VOCAB_SIZE, [END_OF_TEXT])
    return BPETokenizer(vocab, merges, [END_OF_TEXT])

 
def naive_train_bpe(text, vocab_size, special_tokens):

    pretoken_counts = count_pretokens(text, special_tokens)
 
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
 
    words = [list(pretoken) for pretoken in pretoken_counts]
    freqs = list(pretoken_counts.values())
    merges = []
 
    for _ in range(vocab_size - len(vocab)):
        pair_counts = Counter()
        for symbols, freq in zip(words, freqs):
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += freq
        if not pair_counts:
            break
 
        best_count = max(pair_counts.values())
        best_pair = max((p for p, c in pair_counts.items() if c == best_count),
                        key=lambda p: (vocab[p[0]], vocab[p[1]]))
 
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        words = [merge_word(symbols, best_pair, new_id) for symbols in words]
 
    return vocab, merges


def test_train_bpe():
    fast_merges = (train_bpe(DATA_PATH, VOCAB_SIZE, [END_OF_TEXT]))[1]
    slow_merges = (naive_train_bpe(load_text(), VOCAB_SIZE, [END_OF_TEXT]))[1]
 
    if len(fast_merges) == len (slow_merges):
        return True
    else:
        for i, (fast, slow) in enumerate(zip(fast_merges, slow_merges)):
            if fast!=slow:
                return i



def test_round_trip():
    tokenizer = get_tokenizer()
    documents = [d for d in load_text().split(END_OF_TEXT) if d.strip()]
 
    for document in random.Random(0).sample(documents, min(100, len(documents))):

        if tokenizer.decode(tokenizer.encode(document)) == document:
            return True
        else:
            return False

def special_token_to_one_id():
    tokenizer = get_tokenizer()
    eot_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]
 
    ids = tokenizer.encode(f"hello{END_OF_TEXT}world")
    assert ids.count(eot_id) == 1, f"expected exactly one {eot_id}, got {ids}"

def test_id_unt16():
    tokenizer = get_tokenizer()
    ids = tokenizer.encode(load_text())
    if max(ids)<65_536:
        return True
    else:
        return False

def test_identical_runs():
    merges_a = (train_bpe(DATA_PATH, VOCAB_SIZE, [END_OF_TEXT]))[1]
    merges_b = (train_bpe(DATA_PATH, VOCAB_SIZE, [END_OF_TEXT]))[1]
 
    if merges_a == merges_b:
        return True
    else:
        return False

def main():
    print(f"Optimised matches unoptimised {test_train_bpe()}")
    print(f"Round trip test works {test_round_trip()}")
    print(f"Special token goes to one ID {special_token_to_one_id()}")
    print(f"Tesing if fits into numpy {test_id_unt16()}")
    print(f"Testing identical runs {test_identical_runs()}")
    
if __name__ == "__main__":
    main()