
import numpy as np

"""
AI USAGE:
i used ai to look up numpy functions.
i used ai to write the loop ( only this line  while len(vocab) < vocab_size:  cause i am that lazy) and the replace_pair function.
and it added some print satements and cleaned a bit.
"""

vocab_size = 300  # Target vocabulary size


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert to a 1D NumPy array of individual characters
    char_array = np.array(list(text), dtype='U1')

    print(char_array)
    # Output: array(['O', 'n', 'c', 'e', ' ', 'u', 'p', 'o', 'n', ...], dtype='<U1')
    print(f"Shape: {char_array.shape}, Dtype: {char_array.dtype}")
    return char_array


def replace_pair(data, pair):
    merged = pair[0] + pair[1]
    result = []
    i = 0

    while i < len(data):
        if i + 1 < len(data) and data[i] == pair[0] and data[i + 1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(data[i])
            i += 1

    return result


def bpe_encode(data, vocab_size):
    data = [str(token) for token in data] # conv to list 
    vocab = set(data)
    print(f"Init vocabu(size {len(vocab)}): {sorted(vocab)}")

    # count pairs untill vocab is reached
    while len(vocab) < vocab_size: 
        # count the pairs
        pair_counts = {}
        for i in range(len(data) - 1):
            pair = (data[i], data[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            print("No adjacent character pairs found.")
            break

        most_frequent_pair = max(pair_counts, key=pair_counts.get) #get most freq pair
        merged = most_frequent_pair[0] + most_frequent_pair[1] # merge 
        data = replace_pair(data, most_frequent_pair) # replace
        vocab.add(merged) # and add

        print(
            f"Merged {most_frequent_pair} into {merged!r}; "
            f"vocabulary size: {len(vocab)}"
        )

    print(f"Final vocabulary (size {len(vocab)}): {sorted(vocab)}") #done
    return data, vocab


def main():
    print("main")
    print("")
    encoded_data, vocab = bpe_encode(load_data("tsv.txt"), vocab_size)
    print(f"Encoded data length: {len(encoded_data)}")
    print(f"First 50 tokens: {encoded_data[:50]}")


if __name__ == "__main__":
    main()
