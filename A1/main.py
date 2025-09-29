from collections import Counter
import argparse
from typing import Dict, List, Tuple
import random
import math

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
    return normalization_constant

def process_unknown_words(
    linestokens: List[List[str]],
    vocab: List[str] = None,
    threshold: int = 1,
    is_train: bool = True
) -> Tuple[List[List[str]], List[str]]:
    """
    For training:
      - Replace tokens with freq <= threshold with <UNK>, build vocab
    For evaluation:
      - Replace tokens not in vocab with <UNK>
    
    Args:
      linestokens: List of token lists (sentences)
      vocab: vocabulary to enforce for evaluation; ignored if training
      threshold: freq threshold for rare-word replacement in training
      is_train: True if training mode, False for val/test mode
    
    Returns:
      replaced tokens, vocabulary list (for training), or same vocab for evaluation
    """
    if is_train:
        token_freq = Counter()
        for tokens in linestokens:
            token_freq.update(tokens)
        vocab_set = set(tok for tok, freq in token_freq.items() if freq > threshold)
        vocab_set.add('<UNK>')

        replaced_lines = []
        for tokens in linestokens:
            new_tokens = [tok if tok in vocab_set else '<UNK>' for tok in tokens]
            replaced_lines.append(new_tokens)

        return replaced_lines, sorted(vocab_set)
    else:
        if vocab is None:
            raise ValueError("Vocabulary must be provided for evaluation mode")
        vocab_set = set(vocab)
        replaced_lines = []
        for tokens in linestokens:
            new_tokens = [tok if tok in vocab_set else '<UNK>' for tok in tokens]
            replaced_lines.append(new_tokens)
        return replaced_lines, vocab

# ----------------------------
# Byte Pair Encoding (BPE)
# ----------------------------

def _word_to_symbols(word: str) -> Tuple[str, ...]:
    return tuple(list(word) + ['</w>'])

def _get_pair_stats(vocab_symbols: Dict[Tuple[str, ...], int]) -> Counter:
    pairs = Counter()
    for symbols, freq in vocab_symbols.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def _merge_pair_in_word(symbols: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    merged: List[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)

def learn_bpe(word_freqs: Dict[str, int], num_merges: int) -> Dict[Tuple[str, str], int]:
    # Initialize vocabulary as tuples of symbols
    vocab_symbols: Dict[Tuple[str, ...], int] = {
        _word_to_symbols(word): freq for word, freq in word_freqs.items() if freq > 0 and word
    }

    merges_in_order: List[Tuple[str, str]] = []

    for _ in range(num_merges):
        pair_stats = _get_pair_stats(vocab_symbols)
        if not pair_stats:
            break
        best_pair = max(pair_stats.items(), key=lambda kv: kv[1])[0]
        merges_in_order.append(best_pair)

        new_vocab_symbols: Dict[Tuple[str, ...], int] = {}
        for symbols, freq in vocab_symbols.items():
            new_symbols = _merge_pair_in_word(symbols, best_pair)
            new_vocab_symbols[new_symbols] = new_vocab_symbols.get(new_symbols, 0) + freq
        vocab_symbols = new_vocab_symbols

    # Rank merges: lower rank = higher priority
    merges_rank: Dict[Tuple[str, str], int] = {pair: rank for rank, pair in enumerate(merges_in_order)}
    return merges_rank

def bpe_encode_word(word: str, merges_rank: Dict[Tuple[str, str], int]) -> List[str]:
    if not word:
        return []
    symbols = _word_to_symbols(word)
    if len(symbols) == 1:
        return [word]

    def get_pairs(seq: Tuple[str, ...]) -> List[Tuple[str, str]]:
        return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

    pairs = get_pairs(symbols)
    while True:
        candidate_pairs = {p: merges_rank[p] for p in pairs if p in merges_rank}
        if not candidate_pairs:
            break
        best_pair = min(candidate_pairs.items(), key=lambda kv: kv[1])[0]
        symbols = _merge_pair_in_word(symbols, best_pair)
        if len(symbols) == 1:
            break
        pairs = get_pairs(symbols)

    tokens = [s.replace('</w>', '') for s in symbols]
    # Remove empties if any appeared after stripping marker
    return [t for t in tokens if t]

def read_word_frequencies(path: str, lowercase: bool = True) -> Dict[str, int]:
    freqs: Dict[str, int] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if lowercase:
                line = line.lower()
            for word in line.strip().split():
                freqs[word] = freqs.get(word, 0) + 1
    return freqs

def tokenize_line(line: str, use_bpe: bool, merges_rank: Dict[Tuple[str, str], int], lowercase: bool) -> List[str]:
    if lowercase:
        line = line.lower()
    words = line.strip().split()
    if not use_bpe:
        return words
    tokens: List[str] = []
    for w in words:
        tokens.extend(bpe_encode_word(w, merges_rank))
    return tokens


# ----------------------------
# Unigram Language Model
# ----------------------------

def train_unigram_model(linestokens: List[List[str]]) -> Counter:
    """Count unigram occurrences from tokenized lines."""
    unigram_counts = Counter()
    for tokens in linestokens:
        unigram_counts.update(tokens)
    return unigram_counts

def unigram_prob(token: str, unigram_counts: Counter, vocab_size: int, alpha: float) -> float:
    """Calculate smoothed unigram probability with additive smoothing."""
    token_count = unigram_counts.get(token, 0)
    total_tokens = sum(unigram_counts.values())
    return (token_count + alpha) / (total_tokens + alpha * vocab_size)

def corpus_perplexity_unigram(linestokens: List[List[str]], unigram_counts: Counter, vocab_size: int, alpha: float) -> float:
    """Calculate perplexity of unigram model on a tokenized corpus."""
    log_prob_sum = 0.0
    token_count = 0
    for tokens in linestokens:
        for token in tokens:
            p = unigram_prob(token, unigram_counts, vocab_size, alpha)
            log_prob_sum += math.log(p + 1e-12)  # small constant for numerical stability
            token_count += 1
    if token_count == 0:
        return float('inf')
    avg_neg_log_prob = -log_prob_sum / token_count
    return math.exp(avg_neg_log_prob)

def unigram_wrapper(
    train_path: str,
    val_path: str,
    lower: bool,
    use_bpe: bool,
    num_merges: int,
    alpha: float,
    unk_threshold: int = 1
) -> Tuple[float, float]:
    merges_rank = {}
    if use_bpe:
        word_freqs = read_word_frequencies(train_path, lowercase=lower)
        merges_rank = learn_bpe(word_freqs, num_merges=num_merges)

    # Tokenize train and validation data without unknown word replacement
    train_lines = read_tokenized_lines(train_path, use_bpe=use_bpe, merges_rank=merges_rank, lowercase=lower)
    val_lines = read_tokenized_lines(val_path, use_bpe=use_bpe, merges_rank=merges_rank, lowercase=lower)

    # Train unigram counts directly (no unknown word handling)
    unigram_counts_no_unk = train_unigram_model(train_lines)
    vocab_no_unk = list(unigram_counts_no_unk.keys())

    # Calculate perplexity without unknown word handling
    ppl_no_unk = corpus_perplexity_unigram(val_lines, unigram_counts_no_unk, len(vocab_no_unk), alpha)

    # Replace rare words by <UNK> in train, build vocab
    train_replaced, vocab_with_unk = process_unknown_words(train_lines, threshold=unk_threshold, is_train=True)
    # Replace OOV tokens in validation with <UNK>
    val_replaced, _ = process_unknown_words(val_lines, vocab=vocab_with_unk, is_train=False)

    unigram_counts_unk = train_unigram_model(train_replaced)
    # Calculate perplexity with unknown word handling
    ppl_with_unk = corpus_perplexity_unigram(val_replaced, unigram_counts_unk, len(vocab_with_unk), alpha)

    label = 'BPE' if use_bpe else 'whitespace'
    print(f'Perplexity (unigram without unknown-word handling, {label}): {ppl_no_unk:.4f}')
    print(f'Perplexity (unigram with unknown-word handling, {label}): {ppl_with_unk:.4f}')

    return ppl_no_unk, ppl_with_unk



# ----------------------------
# Bigram Language Model utils
# ----------------------------

BOS = '<s>'
EOS = '</s>'

def read_tokenized_lines(path: str, use_bpe: bool, merges_rank: Dict[Tuple[str, str], int], lowercase: bool) -> List[List[str]]:
    lines_tokens: List[List[str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            toks = tokenize_line(line, use_bpe=use_bpe, merges_rank=merges_rank, lowercase=lowercase)
            if not toks:
                continue
            lines_tokens.append([BOS] + toks + [EOS])
    return lines_tokens

def build_vocabulary(lines_tokens: List[List[str]]) -> List[str]:
    vocab_set = set()
    for toks in lines_tokens:
        for t in toks:
            vocab_set.add(t)
    return sorted(vocab_set)

def bigram_wrapper(train_path: str, val_path: str, lower: bool, use_bpe: bool, num_merges: int, alpha: float, unk_threshold: int = None) -> float:
    merges = {}
    if use_bpe:
        word_freqs = read_word_frequencies(train_path, lowercase=lower)
        merges = learn_bpe(word_freqs, num_merges=num_merges)

    train_lines = read_tokenized_lines(train_path, use_bpe=use_bpe, merges_rank=merges, lowercase=lower)
    val_lines   = read_tokenized_lines(val_path,   use_bpe=use_bpe, merges_rank=merges, lowercase=lower)

    if unk_threshold is None:
        bigr, ctx, vocab = train_bigram_model(train_lines)
        ppl = corpus_perplexity(val_lines, bigr, ctx, len(vocab), alpha)
        label = 'BPE' if use_bpe else 'whitespace'
        print(f'Perplexity (bigram, {label}, no UNK, alpha={alpha}): {ppl:.4f}')
        return ppl
    else:
        # unknown word handling
        train_words = [[t for t in s if t not in {BOS, EOS}] for s in train_lines]
        val_words   = [[t for t in s if t not in {BOS, EOS}] for s in val_lines]

        train_u, vocab_u = process_unknown_words(train_words, threshold=unk_threshold, is_train=True)
        vocab_u = sorted(set(vocab_u) | {BOS, EOS})
        val_u, _ = process_unknown_words(val_words, vocab=vocab_u, is_train=False)

        train_u = [[BOS] + s + [EOS] if (not s or s[0] != BOS) else s for s in train_u]
        val_u   = [[BOS] + s + [EOS] if (not s or s[0] != BOS) else s for s in val_u]

        bigr_u, ctx_u, _ = train_bigram_model(train_u)
        ppl_u = corpus_perplexity(val_u, bigr_u, ctx_u, len(vocab_u), alpha)
        label = 'BPE' if use_bpe else 'whitespace'
        print(f'Perplexity (bigram, {label}, UNK<={unk_threshold}, alpha={alpha}): {ppl_u}')
        return ppl_u

def train_bigram_model(lines_tokens: List[List[str]]) -> Tuple[Counter, Counter, List[str]]:
    bigram_counts: Counter = Counter()
    context_counts: Counter = Counter()
    for toks in lines_tokens:
        for i in range(len(toks) - 1):
            h = toks[i]
            w = toks[i + 1]
            bigram_counts[(h, w)] += 1
            context_counts[h] += 1
    vocab = build_vocabulary(lines_tokens)
    return bigram_counts, context_counts, vocab

def bigram_prob(h: str, w: str, bigram_counts: Counter, context_counts: Counter, vocab_size: int, alpha: float) -> float:
    # Additive smoothing
    alpha = float(alpha)
    count_hw = bigram_counts.get((h, w), 0)
    count_h = context_counts.get(h, 0)

    # Avoid division by zero if h is unseen and alpha=0
    if (count_h + alpha * vocab_size) == 0:
        return 0.0
    return (count_hw + alpha) / (count_h + alpha * vocab_size)

def corpus_perplexity(lines_tokens: List[List[str]], bigram_counts: Counter, context_counts: Counter, vocab_size: int, alpha: float) -> float:
    log_prob_sum = 0.0
    token_count = 0
    for toks in lines_tokens:
        for i in range(len(toks) - 1):
            h = toks[i]
            w = toks[i + 1]
            p = bigram_prob(h, w, bigram_counts, context_counts, vocab_size, alpha)
            log_prob_sum += math.log(p + 1e-12)
            token_count += 1
    if token_count == 0:
        return float('inf')
    avg_neg_log = -log_prob_sum / token_count
    return float(math.exp(avg_neg_log))

def next_token_distribution(history_token: str, bigram_counts: Counter, context_counts: Counter, vocab: List[str], alpha: float, exclude_tokens: List[str]) -> List[Tuple[str, float]]:
    denom = context_counts.get(history_token, 0) + alpha * len(vocab)
    dist: List[Tuple[str, float]] = []
    for w in vocab:
        if w in exclude_tokens:
            continue
        p = (bigram_counts.get((history_token, w), 0) + alpha) / denom
        dist.append((w, p))
    dist.sort(key=lambda x: x[1], reverse=True)
    return dist

def sample_next_token(history_token: str, bigram_counts: Counter, context_counts: Counter, vocab: List[str], alpha: float, temperature: float, top_k: int, rng: random.Random, exclude_tokens: List[str]) -> str:
    candidates = next_token_distribution(history_token, bigram_counts, context_counts, vocab, alpha, exclude_tokens)
    if top_k > 0 and top_k < len(candidates):
        candidates = candidates[:top_k]
    # Apply temperature by exponentiating probabilities
    if temperature <= 0:
        temperature = 1.0
    reweighted = [(w, (p ** (1.0 / temperature))) for w, p in candidates]
    total = sum(p for _, p in reweighted)
    if total == 0:
        # fallback uniform
        choices = [w for w, _ in candidates]
        return rng.choice(choices)
    probs = []
    acc = 0.0
    for _, p in reweighted:
        acc += p / total
        probs.append(acc)
    r = rng.random()
    for (w, _), cutoff in zip(candidates, probs):
        if r <= cutoff:
            return w
    return candidates[-1][0]

def detokenize(tokens: List[str], use_bpe: bool) -> str:
    if not tokens:
        return ''
    if use_bpe:
        # Best-effort: space-join BPE tokens (readability over correctness)
        return ' '.join(tokens)
    return ' '.join(tokens)

def interactive_session(bigram_counts: Counter, context_counts: Counter, vocab: List[str], merges_rank: Dict[Tuple[str, str], int], use_bpe: bool, lowercase: bool, alpha: float, max_len: int, temperature: float, top_k: int, seed: int) -> None:
    rng = random.Random(seed)
    print("Interactive bigram LM. Type text and press Enter. Commands: /quit, /help")
    print(f"Mode: {'BPE' if use_bpe else 'whitespace'} | alpha={alpha} temp={temperature} top_k={top_k} max_len={max_len}")
    while True:
        try:
            user_in = input('prompt> ').strip()
        except EOFError:
            break
        if not user_in:
            continue
        if user_in.lower() in {'/q', '/quit', 'exit'}:
            break
        if user_in.lower() in {'/h', '/help'}:
            print("Enter a prompt. We'll continue generating tokens until </s> or max_len.")
            print("Commands: /quit to exit, /help to see this message again.")
            continue

        # Tokenize prompt
        toks = tokenize_line(user_in, use_bpe=use_bpe, merges_rank=merges_rank, lowercase=lowercase)
        history = [BOS] + toks

        # Show top-10 next-token suggestions for the last token
        dist = next_token_distribution(history[-1], bigram_counts, context_counts, vocab, alpha, exclude_tokens=[BOS])
        print('Top next tokens: ' + ', '.join([f"{w}:{p:.3f}" for w, p in dist[:10]]))

        # Generate continuation
        generated: List[str] = []
        h = history[-1]
        for _ in range(max_len):
            w = sample_next_token(h, bigram_counts, context_counts, vocab, alpha, temperature, top_k, rng, exclude_tokens=[BOS])
            if w == EOS:
                break
            generated.append(w)
            h = w

        print('Generated tokens: ' + ' '.join(generated) if generated else '(no tokens)')
        print('Generated text:   ' + detokenize(generated, use_bpe=use_bpe))

# ----------------------------
# Grid evaluation
# ----------------------------

def run_simple_grid(train_path: str, val_path: str, lower: bool, num_merges: int, alphas: List[float], unk_thresholds: List[int], compare_bpe: bool):
    lines = []
    for use_bpe in ([False, True] if compare_bpe else [False]):
        # no unknowns handling
        for a in alphas:
            ppl = bigram_wrapper(train_path, val_path, lower, use_bpe, num_merges, a, unk_threshold=None)
            lines.append(['bigram', 'no_unk', 'BPE' if use_bpe else 'whitespace', a, '-', f'{ppl:.4f}'])
        # with unknown handling
        for thr in unk_thresholds:
            for a in alphas:
                ppl = bigram_wrapper(train_path, val_path, lower, use_bpe, num_merges, a, unk_threshold=thr)
                lines.append(['bigram', 'with_unk', 'BPE' if use_bpe else 'whitespace', a, thr, f'{ppl:.4f}'])

    print("\n\n=== Bigram eval results ===\n")

    headers = ["model", "unk", "tokenization", "alpha", "unk_threshold", "ppl"]
    rows = [headers] + lines
    colwidths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    for r in rows:
        print(" | ".join(str(val).ljust(colwidths[i]) for i, val in enumerate(r)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N-gram model with optional BPE tokenization')
    parser.add_argument('--train_path', type=str, default='./A1_DATASET/train.txt')
    parser.add_argument('--val_path', type=str, default='./A1_DATASET/val.txt')
    parser.add_argument('-n', '--n', type=int, default=1, help='N for n-grams')
    parser.add_argument('--use_bpe', action='store_true', help='Enable BPE tokenization')
    parser.add_argument('--num_merges', type=int, default=1000, help='Number of BPE merges to learn')
    parser.add_argument('--compare_bpe', action='store_true', help='Train/evaluate bigram models with and without BPE and compare')
    parser.add_argument('--alpha', type=float, default=1.0, help='Additive smoothing for bigram model')
    parser.add_argument('--repl', action='store_true', help='Start interactive session with the bigram model')
    parser.add_argument('--max_len', type=int, default=30, help='Max generated tokens in REPL')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature in REPL')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling in REPL (0 disables)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for REPL sampling')
    parser.add_argument('--no-lower', dest='lower', action='store_false', help='Disable lowercasing')

    # Evaluation
    parser.add_argument('--run_simple_grid', action='store_true', help='Run grid of bigram experiments and print table')
    parser.add_argument('--alphas', type=str, default='0,0.5,1.0', help='Comma-separated add-k values')
    parser.add_argument('--unk_thresholds', type=str, default='1,2', help='Comma-separated UNK thresholds')

    parser.set_defaults(lower=True)
    args = parser.parse_args()

    train_path = args.train_path
    val_path = args.val_path
    n_value = args.n
    use_bpe = args.use_bpe
    lower = args.lower
    alphas = [float(x.strip()) for x in args.alphas.split(',') if x.strip() != '']
    unk_thresholds = [int(x.strip()) for x in args.unk_thresholds.split(',') if x.strip() != '']

    if args.run_simple_grid:
        run_simple_grid(
            train_path=args.train_path,
            val_path=args.val_path,
            lower=args.lower,
            num_merges=args.num_merges,
            alphas=alphas,
            unk_thresholds=unk_thresholds,
            compare_bpe=args.compare_bpe
        )
    
    else:
        # Unigram model
        if n_value == 1:
            if args.compare_bpe:
                # Compare unigram models with and without BPE
                unigram_wrapper(train_path, val_path, lower, use_bpe=False, num_merges=args.num_merges, alpha=args.alpha)
                unigram_wrapper(train_path, val_path, lower, use_bpe=True, num_merges=args.num_merges, alpha=args.alpha)
            else:
                unigram_wrapper(train_path, val_path, lower, use_bpe, args.num_merges, args.alpha)
            exit(0)
        elif n_value == 2:
            pass

        # Bigram model with BPE comparison mode
        # Comparison mode: always build bigram models with and without BPE and report perplexity
        if args.compare_bpe:
            # Learn BPE merges on training data
            word_freqs = read_word_frequencies(train_path, lowercase=lower)
            bpe_merges = learn_bpe(word_freqs, num_merges=args.num_merges)

            # Tokenize corpora
            train_plain = read_tokenized_lines(train_path, use_bpe=False, merges_rank={}, lowercase=lower)
            val_plain = read_tokenized_lines(val_path, use_bpe=False, merges_rank={}, lowercase=lower)

            train_bpe = read_tokenized_lines(train_path, use_bpe=True, merges_rank=bpe_merges, lowercase=lower)
            val_bpe = read_tokenized_lines(val_path, use_bpe=True, merges_rank=bpe_merges, lowercase=lower)

            # Train
            bigram_plain, context_plain, vocab_plain = train_bigram_model(train_plain)
            bigram_bpe, context_bpe, vocab_bpe = train_bigram_model(train_bpe)

            # Eval
            ppl_plain = corpus_perplexity(val_plain, bigram_plain, context_plain, len(vocab_plain), args.alpha)
            ppl_bpe = corpus_perplexity(val_bpe, bigram_bpe, context_bpe, len(vocab_bpe), args.alpha)

            print(f'Perplexity (bigram, whitespace): {ppl_plain:.4f}')
            print(f'Perplexity (bigram, BPE): {ppl_bpe:.4f}')
            if args.repl:
                print('\nLaunching REPL with the BPE model (since we learned merges already).')
                interactive_session(bigram_bpe, context_bpe, vocab_bpe, merges_rank=bpe_merges, use_bpe=True, lowercase=lower, alpha=args.alpha, max_len=args.max_len, temperature=args.temperature, top_k=args.top_k, seed=args.seed)
        else:
            # Preserve original behavior for non-bigram n
            if n_value != 2:
                merges_rank: Dict[Tuple[str, str], int] = {}
                if use_bpe:
                    word_freqs = read_word_frequencies(train_path, lowercase=lower)
                    merges_rank = learn_bpe(word_freqs, num_merges=args.num_merges)

                n_grams: List[str] = []
                with open(train_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        tokens = tokenize_line(line, use_bpe=use_bpe, merges_rank=merges_rank, lowercase=lower)
                        n_grams.extend(n_gram(tokens, n_value))
                counter = Counter(n_grams)
                print(len(counter.keys()))
            else:
                # Single bigram model path according to --use_bpe
                merges_rank: Dict[Tuple[str, str], int] = {}
                if use_bpe:
                    word_freqs = read_word_frequencies(train_path, lowercase=lower)
                    merges_rank = learn_bpe(word_freqs, num_merges=args.num_merges)

                train_lines = read_tokenized_lines(train_path, use_bpe=use_bpe, merges_rank=merges_rank, lowercase=lower)
                val_lines = read_tokenized_lines(val_path, use_bpe=use_bpe, merges_rank=merges_rank, lowercase=lower)

                bigram_counts, context_counts, vocab = train_bigram_model(train_lines)
                ppl = corpus_perplexity(val_lines, bigram_counts, context_counts, len(vocab), args.alpha)
                label = 'BPE' if use_bpe else 'whitespace'
                print(f'Perplexity (bigram, {label}): {ppl:.4f}')
                if args.repl:
                    interactive_session(bigram_counts, context_counts, vocab, merges_rank=merges_rank, use_bpe=use_bpe, lowercase=lower, alpha=args.alpha, max_len=args.max_len, temperature=args.temperature, top_k=args.top_k, seed=args.seed)
