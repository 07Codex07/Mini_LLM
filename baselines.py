import numpy as np
from collections import Counter

import bpe
import testTransformerModel as tm  # loads the model + tokenizer

# ==========================
# Same tokens the transformer sees
# ==========================

with open(tm.CORPUS_FILE, "r", encoding="utf-8") as f:
    text = f.read()

words = tm.tokenize_words(text)

# same as tm.encode but each unique word only once, much faster
word_cache = {w: [tm.token_to_id[t] for t in bpe.apply_bpe_to_word(w, tm.merges_rank) if t in tm.token_to_id]
              for w in set(words)}
tokens = np.array([t for w in words for t in word_cache[w]], dtype=np.int32)
V = tm.vocabSize

# test = the same last 5% that training holds out, so nothing here has seen it.
# older checkpoints don't save val_fraction and were trained on everything
CLEAN_SPLIT = "val_fraction" in tm.cfg
n = len(tokens)
n_test = int(n * tm.cfg.get("val_fraction", 0.05))
train = tokens[:n - 2 * n_test]
dev = tokens[n - 2 * n_test:n - n_test]   # for tuning the n-gram weights
test = tokens[n - n_test:]

print(f"\nCorpus: {tm.CORPUS_FILE} | {len(words)} words -> {n} BPE tokens (vocab {V})")
print(f"Split: train {len(train)} / dev {len(dev)} / test {len(test)} tokens (contiguous)")

# ==========================
# N-gram baselines (interpolated: trigram + bigram + add-1 unigram)
# ==========================

uni = np.bincount(train, minlength=V).astype(np.float64)
p_uni = (uni + 1) / (uni.sum() + V)

bi = Counter(zip(train[:-1].tolist(), train[1:].tolist()))
tri = Counter(zip(train[:-2].tolist(), train[1:-1].tolist(), train[2:].tolist()))
bi_ctx = Counter(train[:-1].tolist())
tri_ctx = Counter(zip(train[:-2].tolist(), train[1:-1].tolist()))


def ngram_probs(seq):
    # p_uni, p_bi, p_tri of each token given the 2 before it
    seq = seq.tolist()
    out = np.zeros((len(seq) - 2, 3))
    for i in range(2, len(seq)):
        a, b, c = seq[i - 2], seq[i - 1], seq[i]
        out[i - 2, 0] = p_uni[c]
        out[i - 2, 1] = bi[(b, c)] / bi_ctx[b] if bi_ctx[b] else 0.0
        out[i - 2, 2] = tri[(a, b, c)] / tri_ctx[(a, b)] if tri_ctx[(a, b)] else 0.0
    return out


def ppl(p):
    return float(np.exp(-np.mean(np.log(p))))


dev_p, test_p = ngram_probs(dev), ngram_probs(test)

# pick interpolation weights on dev, then score test
best_bi = min(((l,) for l in np.arange(0.05, 1.0, 0.05)),
              key=lambda l: ppl(l[0] * dev_p[:, 1] + (1 - l[0]) * dev_p[:, 0]))
best_tri = min(((l3, l2) for l3 in np.arange(0.05, 1.0, 0.05) for l2 in np.arange(0.05, 1.0, 0.05) if l3 + l2 < 1),
               key=lambda l: ppl(l[0] * dev_p[:, 2] + l[1] * dev_p[:, 1] + (1 - l[0] - l[1]) * dev_p[:, 0]))

results = {
    "Uniform": float(V),
    "Unigram": ppl(test_p[:, 0]),
    "Bigram (interp.)": ppl(best_bi[0] * test_p[:, 1] + (1 - best_bi[0]) * test_p[:, 0]),
    "Trigram (interp.)": ppl(best_tri[0] * test_p[:, 2] + best_tri[1] * test_p[:, 1]
                             + (1 - sum(best_tri)) * test_p[:, 0]),
}

# ==========================
# Transformer on the same test tokens
# ==========================


def forward_all_positions(ctx_batch):
    return tm.softmax(tm.hidden_states(ctx_batch) @ tm.input_embeddings[:V].T)   # (B, seq_len, V)


S = tm.seq_len
n_win = (len(test) - 1) // S
windows = test[:n_win * S + 1]
ctx = np.stack([windows[i * S:(i + 1) * S] for i in range(n_win)])
tgt = np.stack([windows[i * S + 1:(i + 1) * S + 1] for i in range(n_win)])

p_all = []
for b in range(0, n_win, 256):
    probs = forward_all_positions(ctx[b:b + 256])
    B = probs.shape[0]
    p_all.append(probs[np.arange(B)[:, None], np.arange(S)[None, :], tgt[b:b + 256]])
p_all = np.concatenate(p_all)                         # (n_win, S)

results["Transformer (all positions)"] = ppl(p_all + 1e-10)
results[f"Transformer (last position, {S} tokens of context)"] = ppl(p_all[:, -1] + 1e-10)

# ==========================
# Report
# ==========================

print(f"\nTest perplexity (lower is better), interpolation weights: bigram {best_bi[0]:.2f}, "
      f"trigram {best_tri[0]:.2f}/{best_tri[1]:.2f}\n")
print(f"| {'Model':<50} | {'Perplexity':>10} | {'Loss (nats)':>11} |")
print(f"|{'-' * 52}|{'-' * 12}|{'-' * 13}|")
for name, p in results.items():
    print(f"| {name:<50} | {p:>10.1f} | {np.log(p):>11.3f} |")

if CLEAN_SPLIT:
    print("\nNo model saw the test slice during training: it is the held-out tail of the corpus.")
else:
    print("\nNOTE: the n-grams never saw the test slice. This (older) transformer checkpoint was trained on")
    print("random windows from the WHOLE corpus, so its test numbers are optimistic (it has seen this text).")
    print("For a clean comparison, retrain with the contiguous split in transormerBlock.py.")
