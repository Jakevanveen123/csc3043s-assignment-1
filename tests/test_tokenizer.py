from collections import Counter
import sys
from pathlib import Path
 
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


def test_BPE_tokenizer():
    return None
def test_round_trip():
    return None

def test_end_of_text():
    return None

def test_id_unt16():
    return None

def test_identical_runs():
    return None

def main():
    print(test_train_bpe())

if __name__ == "__main__":
    main()