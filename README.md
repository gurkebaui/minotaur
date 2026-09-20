# Minotaur

A small learning project for understanding the fundamentals of machine learning and NumPy.

## What this is

`bpe.py` is a simple NumPy-based Byte Pair Encoding (BPE) experiment. It reads text, starts with a character-level vocabulary, repeatedly finds the most frequent adjacent pair, merges that pair, and expands the vocabulary until it reaches 300 entries.

The implementation intentionally favors readability and simplicity over speed. It is a hands-on exercise in:

- Loading and representing text data
- Using NumPy arrays
- Counting token pairs
- Building a vocabulary
- Understanding a basic tokenization algorithm

## Running it

Install NumPy, then run:

```bash
python bpe.py
```

The script reads `tsv.txt` from the project directory and prints vocabulary and encoding progress.

## Where this is going

This project is a NumPy trial and a stepping stone toward deeper machine-learning work. Future versions will focus more on PyTorch, neural networks, and practical ML experiments.


## Dataset

I used the training split of  roneneldan/TinyStories dataset downloaded as a txt

## AI usage

AI usage is labled in the code. 
