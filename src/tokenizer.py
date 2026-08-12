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

def count_pretokens(text,special_tokens=(END_OF_TEXT,)):

    counts = Counter()
    split_pattern = "|".join(re.escape(s) for s in special_tokens)
    for document in re.split(split_pattern, text):
        counts.update(pretokenize(document))
    return counts

def merge_word(symbols, pair, new_id):

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

def train_bpe(input_path, vocab_size, special_tokens, max_chars=300_000_000,
              verbose=False):

    with open(input_path, encoding="utf-8") as f:
        text = f.read(max_chars)

    pretoken_counts = count_pretokens(text, special_tokens)
    del text     

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    
    num_merges = vocab_size - len(vocab)

    if num_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} is smaller than the {len(vocab)} "
                         f"special tokens + characters")


    words = [list(pretoken) for pretoken in pretoken_counts]
    freqs = list(pretoken_counts.values())

    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for i in range(len(words)):
        symbols = words[i]
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += freqs[i]
            pair_to_words[pair].add(i)

    merges = []
    start = time.time()

    for step in range(num_merges):
        if not pair_counts:
            break

        best_count = max(pair_counts.values())
        best_pair = max((p for p, c in pair_counts.items() if c == best_count),
                        key=lambda p: (vocab[p[0]], vocab[p[1]]))


        left, right = best_pair
        new_id = len(vocab)
        vocab[new_id] = vocab[left] + vocab[right]
        merges.append((vocab[left], vocab[right]))

        affected = pair_to_words.pop(best_pair, set())
        pair_counts.pop(best_pair, None)

        for i in affected:
            old_symbols = words[i]
            new_symbols = merge_word(old_symbols, best_pair, new_id)
            words[i] = new_symbols
            freq = freqs[i]


            old_pairs = Counter(zip(old_symbols, old_symbols[1:]))
            new_pairs = Counter(zip(new_symbols, new_symbols[1:]))

            for pair in set(old_pairs) | set(new_pairs):
                delta = new_pairs[pair] - old_pairs[pair]
                if delta != 0:
                    pair_counts[pair] += delta * freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                        pair_to_words.pop(pair, None)
                if new_pairs[pair] == 0:
                    holders = pair_to_words.get(pair)
                    if holders is not None:
                        holders.discard(i)
                elif pair in pair_counts:
                    pair_to_words[pair].add(i)

        
        #This if statement was copy pasted from the assignment
        if verbose and (step + 1) % 500 == 0:
            print(f"  merge {step + 1:5d}: {vocab[left]!r} + {vocab[right]!r} -> "
                  f"{vocab[new_id]!r} (count {best_count:,})   "
                  f"[{time.time() - start:.1f}s]")

    return vocab, merges
