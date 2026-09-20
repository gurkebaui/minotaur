
import numpy as np
vocab_size = 1000  # Example vocabulary size, can be adjusted as needed


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert to a 1D NumPy array of individual characters
    char_array = np.array(list(text), dtype='U1')

    print(char_array)
    # Output: array(['O', 'n', 'c', 'e', ' ', 'u', 'p', 'o', 'n', ...], dtype='<U1')
    print(f"Shape: {char_array.shape}, Dtype: {char_array.dtype}")
    return char_array


def bpe_encode(data, vocab_size):
    vocab = np.unique(data)
    vocab = vocab[:vocab_size]  # Limit the vocabulary size
    print(f"Vocabulary (size {len(vocab)}): {vocab}")

    pair_counts = {}
    for i in range(len(data) - 1):
        pair = (data[i], data[i + 1])
        if pair in pair_counts:
            pair_counts[pair] += 1
        else:
            pair_counts[pair] = 1

    print(f"Pair counts (size {len(pair_counts)}): {pair_counts}")

    if not pair_counts:
        print("No adjacent character pairs found; input must contain at least two characters.")
        return

    most_frequent_pair = max(pair_counts, key=pair_counts.get)
    print(f"Most frequent pair: {most_frequent_pair} with count {pair_counts[most_frequent_pair]}")





def main():
    print("main")
    print("")
    bpe_encode(load_data("tsv.txt"), vocab_size)

if __name__ == "__main__":
    main()
