import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
 
from src.tokenizer import END_OF_TEXT, BPETokenizer, train_bpe
TRAIN = "/content/drive/MyDrive/CSC3043S/train.txt"
VALID_FULL = "data/TinyStoriesV2-GPT4-valid.txt"
VALID = "/content/drive/MyDrive/CSC3043S/valid.txt"
TEST = "data/test.txt"
 
VOCAB_SIZES = [4000, 1000]

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

def build(vocab_size):

    vocab, merges = train_bpe(
        TRAIN,
        vocab_size,
        [END_OF_TEXT],
        verbose=True
    )
    tokenizer = BPETokenizer(vocab, merges, [END_OF_TEXT])
    tokenizer.save(f"data/tok{vocab_size}_vocab.json",
                   f"data/tok{vocab_size}_merges.json")
 
    for name, path in [("train", TRAIN), ("valid", VALID), ("test", TEST)]:
        output = f"data/{name}_{vocab_size}.npy"
        print(f"  {output}: {encode_file(path, tokenizer, output):,} tokens")


if __name__ == "__main__":
    for size in VOCAB_SIZES:
        build(size)