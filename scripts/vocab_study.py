from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.tokenizer import BPETokenizer
from src.tokenizer import train_bpe

END_OF_TEXT = "<|endoftext|>"

VALID_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
SIZES = [1000, 2000, 4000, 8000, 16000]

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

if __name__ == "__main__":
    vocab_size_study(VALID_PATH, SIZES)
    