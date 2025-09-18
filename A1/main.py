from collections import Counter

def n_gram(tokens, n):
    n_grams = [' '.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    return n_grams

def normalization_constant(counter, token, n):
    if n > 1:
        normalization_constant = 0
        for bigram in counter.keys():
            if token == bigram.split()[0]:
                normalization_constant += counter[bigram]
    else: normalization_constant = sum(counter.values())

if __name__ == '__main__':
    train_path = './A1_DATASET/train.txt'
    val_path  = './A1_DATASET/val.txt'

    n = 1
    n_grams = []

    with open(train_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.lower().split()
            n_grams.extend(n_gram(tokens, n))
    
    counter = Counter(n_grams)

    print(len(counter.keys()))
