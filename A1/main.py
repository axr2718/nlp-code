from collections import Counter

def n_gram(tokens, n):
    n_grams = [' '.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    return n_grams

if __name__ == '__main__':
    train_path = './A1_DATASET/train.txt'
    val_path  = './A1_DATASET/val.txt'

    n = 5
    n_grams = []

    with open(train_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.lower().split()
            n_grams.extend(n_gram(tokens, n))

    #print(n_grams)
    
    counter = Counter(n_grams)

    print(counter)
