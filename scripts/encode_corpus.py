import numpy as np

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