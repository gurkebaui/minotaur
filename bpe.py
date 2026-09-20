import numpy as np
vocab_size = 300  # Target vocabulary size


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: #loads data from text as np arr of uni char
        text = f.read()

    return np.fromiter(text, dtype='U1', count=len(text))


def bpe_encode(data, vocab_size):
    characters = np.unique(data) #every unic char
    tokens = np.searchsorted(characters, data).astype(np.int32, copy=False)# convert into int arr for the uni
    id_to_token = characters.tolist() 
    vocab = set(id_to_token) 
    print(f"Initial vocabulary (size {len(vocab)}): {sorted(vocab)}")

    # perform BPE magic
    while len(vocab) < vocab_size and tokens.size > 1:
        base = int(tokens.max()) + 1
        pair_ids = tokens[:-1].astype(np.int64) * base + tokens[1:]
        pair_counts = np.bincount(pair_ids)
        present_pair_ids = np.flatnonzero(pair_counts)

        if present_pair_ids.size == 0:
            break

        most_frequent_pair_id = present_pair_ids[
            np.argmax(pair_counts[present_pair_ids])
        ]
        left_id = int(most_frequent_pair_id // base)
        right_id = int(most_frequent_pair_id % base)
        matches = (tokens[:-1] == left_id) & (tokens[1:] == right_id)
        starts = np.flatnonzero(matches)
        starts = starts[np.diff(np.r_[-1, starts]) > 1]

        merged = id_to_token[left_id] + id_to_token[right_id]
        remove_positions = starts + 1
        remove_mask = np.zeros(tokens.size, dtype=bool)
        remove_mask[remove_positions] = True
        merged_tokens = tokens[~remove_mask]
        merged_tokens[starts - np.arange(starts.size)] = len(id_to_token)

        tokens = merged_tokens
        id_to_token.append(merged)
        vocab.add(merged)

        print(
            f"Merged {id_to_token[left_id]!r} + {id_to_token[right_id]!r} "
            f"into {merged!r}; vocabulary size: {len(vocab)}"
        )

    print(f"Final vocabulary (size {len(vocab)}): {sorted(vocab)}")
    return tokens, vocab, id_to_token
    """ 
    Now, i let ai clean this up but damn 
    it just roasted my code by changing all of it. valid tho, mine was slow af, but i should have committed to gh before. my mistake.
    """

def main():
    # main 
    print("main")
    print("")
    encoded_data, vocab, id_to_token = bpe_encode(load_data("tsv.txt"), vocab_size)
    print(f"Encoded data length: {len(encoded_data)}")
    print(f"First 50 tokens: {[id_to_token[i] for i in encoded_data[:50]]}")

if __name__ == "__main__":
    main()