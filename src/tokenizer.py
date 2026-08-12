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

class BPETokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens else []
        self.token_to_id = {token: i for i, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._cache = {}
        self._special_re = re.compile(
            "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        ) if self.special_tokens else None

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, encoding="utf-8") as f:
            vocab = {int(i): token.encode("latin-1") for i, token in json.load(f).items()}
        with open(merges_path, encoding="utf-8") as f:
            merges = [(left.encode("latin-1"), right.encode("latin-1"))
                      for left, right in json.load(f)]
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_path, merges_path):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({str(i): token.decode("latin-1") for i, token in self.vocab.items()},
                      f, ensure_ascii=False)
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump([[left.decode("latin-1"), right.decode("latin-1")]
                       for left, right in self.merges], f, ensure_ascii=False)

    def _apply_merges(self, symbols):
        symbols = list(symbols)
        while len(symbols) > 1:
            best_rank, best_index = None, None
            for i, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, i
            if best_index is None:
                break
            symbols[best_index:best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]
        return symbols

    def _encode_pretoken(self, pretoken):
        if pretoken not in self._cache:
            symbols = self._apply_merges([bytes([b]) for b in pretoken])
            self._cache[pretoken] = [self.token_to_id[s] for s in symbols]
        return self._cache[pretoken]

    def encode(self, text):
        ids = []
        pieces = self._special_re.split(text) if self._special_re else [text]
        for piece in pieces:
            if piece in self.special_tokens:
                ids.append(self.token_to_id[piece.encode("utf-8")])
            else:
                for pretoken in pretokenize(piece):
                    ids.extend(self._encode_pretoken(pretoken))
        return ids

    def encode_iterable(self, iterable):

        buffer = ""
        for chunk in iterable:
            buffer += chunk
            if len(buffer) < (1 << 20):
                continue
            cut = buffer.rfind("\n")
            while cut > 0 and buffer[cut - 1].isspace():
                cut = buffer.rfind("\n", 0, cut)
            if cut > 0:
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids):
        tokens = [self.vocab[i] for i in ids]
        raw = b"".join(tokens)
        return raw.decode("utf-8", errors="replace")


def vocab_size_study(input_path, vocab_sizes, special_tokens=(END_OF_TEXT,)):

    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    n_bytes = len(text.encode("utf-8"))
    n_chars = len(text)

    results = []
    print(f"{'vocab':>8} {'tokens':>14} {'bytes/token':>12} {'chars/token':>12}")
    for vocab_size in vocab_sizes:
        vocab, merges = train_bpe(input_path, vocab_size, list(special_tokens))
        n_tokens = len(BPETokenizer(vocab, merges, special_tokens).encode(text))
        results.append((vocab_size, n_bytes / n_tokens, n_chars / n_tokens))
        print(f"{vocab_size:>8,} {n_tokens:>14,} "
              f"{n_bytes / n_tokens:>12.3f} {n_chars / n_tokens:>12.3f}")
    return results

def encode_file(input_path, tokenizer, output_path, block_chars=1 << 22):

    def blocks(handle):
        while True:
            data = handle.read(block_chars)
            if not data:
                return
            yield data

    parts, buffer = [], []
    with open(input_path, encoding="utf-8", errors="replace") as f:
        for token_id in tokenizer.encode_iterable(blocks(f)):
            buffer.append(token_id)
            if len(buffer) >= 1_000_000:
                parts.append(np.array(buffer, dtype=np.uint16))
                buffer = []
    parts.append(np.array(buffer, dtype=np.uint16))

    ids = np.concatenate(parts)
    np.save(output_path, ids)
    return len(ids)