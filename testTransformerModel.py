import numpy as np
import os
import pickle
import re

import bpe

WEIGHTS_SUFFIX = "_best"  # "_best" = lowest val loss, "" = last epoch

# ==========================
# Load trained weights + config
# ==========================

with open("bpe_data.pkl", "rb") as f:
    bpe_data = pickle.load(f)
merges = bpe_data["merges"]
token_to_id = bpe_data["token_to_id"]
id_to_token = bpe_data["id_to_token"]
merges_rank = bpe.build_merges_rank(merges)

vocabSize = len(token_to_id)
PAD_ID = vocabSize

with open("model_config.pkl", "rb") as f:
    cfg = pickle.load(f)
seq_len = cfg["seq_len"]
embeddingsDim = cfg["embeddingsDim"]
num_heads = cfg["num_heads"]
d_head = embeddingsDim // num_heads

if not os.path.exists(f"transformer_embeddings{WEIGHTS_SUFFIX}.npy"):
    WEIGHTS_SUFFIX = ""
input_embeddings = np.load(f"transformer_embeddings{WEIGHTS_SUFFIX}.npy")
EMBED_SCALE = np.sqrt(embeddingsDim)

# load by index - glob("transformer_layer*.npz") also picked up the _best files and ran 2 layers as 4.
# old configs don't have num_layers so just count the files
num_layers = cfg.get("num_layers")
if num_layers is None:
    num_layers = 0
    while os.path.exists(f"transformer_layer{num_layers}{WEIGHTS_SUFFIX}.npz"):
        num_layers += 1
layers = [dict(np.load(f"transformer_layer{i}{WEIGHTS_SUFFIX}.npz")) for i in range(num_layers)]

# old checkpoints are post-LN + relu, newer ones save this in the config
PRE_LN = cfg.get("pre_ln", False)
ACTIVATION = cfg.get("activation", "relu")
final_ln = dict(np.load(f"transformer_final_ln{WEIGHTS_SUFFIX}.npz")) if PRE_LN else None

# old checkpoints were trained on lowercase letters only. if the input isn't cleaned the same way,
# anything not in the vocab gets dropped ("The" -> "he"). guess from the vocab if config doesn't say
LOWERCASE_ONLY = cfg.get("lowercase_only",
                         not any(not ch.isalpha() or ch.isupper()
                                 for tok in token_to_id for ch in tok.replace(bpe.END, "")))
CORPUS_FILE = cfg.get("corpus_file", "corpus.txt" if LOWERCASE_ONLY else "corpus_v2_clean.txt")

print(f"Loaded {num_layers} layer(s){' (best checkpoint)' if WEIGHTS_SUFFIX else ''}, "
      f"embeddingsDim={embeddingsDim}, num_heads={num_heads}, BPE vocabSize={vocabSize}, "
      f"lowercase_only={LOWERCASE_ONLY}, {'pre-LN' if PRE_LN else 'post-LN'}/{ACTIVATION}")

causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def build_positional_encoding(seq_len, dim):
    matrix = np.zeros((seq_len, dim))
    for pos in range(seq_len):
        for i in range(dim):
            angle = pos / (10000 ** ((2 * (i // 2)) / dim))
            matrix[pos, i] = np.sin(angle) if i % 2 == 0 else np.cos(angle)
    return matrix


pos_encoding = build_positional_encoding(seq_len, embeddingsDim)[None, :, :]

# ==========================
# Forward pass only (same math as training)
# ==========================


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def split_heads(t, B):
    return t.reshape(B, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)


def merge_heads(t, B):
    return t.transpose(0, 2, 1, 3).reshape(B, seq_len, embeddingsDim)


def attention(X, layer):
    B = X.shape[0]
    Q, K, V = X @ layer['W_Q'], X @ layer['W_K'], X @ layer['W_V']
    Qh, Kh, Vh = split_heads(Q, B), split_heads(K, B), split_heads(V, B)
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
    scores = np.where(causal_mask, -1e9, scores)
    attn_w = softmax(scores)
    context = merge_heads(attn_w @ Vh, B)
    return context @ layer['W_O_attn']


def ffn(X, layer):
    pre = X @ layer['W_ff1']
    if ACTIVATION == "gelu":
        h = 0.5 * pre * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (pre + 0.044715 * pre ** 3)))
    else:
        h = np.maximum(0, pre)
    return h @ layer['W_ff2']


def block(X, layer):
    if PRE_LN:
        X = X + attention(layernorm(X, layer['gamma1'], layer['beta1']), layer)
        return X + ffn(layernorm(X, layer['gamma2'], layer['beta2']), layer)
    X1 = layernorm(X + attention(X, layer), layer['gamma1'], layer['beta1'])
    X2 = layernorm(X1 + ffn(X1, layer), layer['gamma2'], layer['beta2'])
    return X2


def hidden_states(context_ids_batch):
    # final hidden state at every position, (B, seq_len, dim)
    idx = np.array(context_ids_batch)   # (B, seq_len)
    X = input_embeddings[idx] * EMBED_SCALE + pos_encoding
    for layer in layers:
        X = block(X, layer)
    if PRE_LN:
        X = layernorm(X, final_ln['gamma_f'], final_ln['beta_f'])
    return X


def forward(context_ids_batch):
    last = hidden_states(context_ids_batch)[:, -1, :]
    E_real = input_embeddings[:vocabSize]
    logits = last @ E_real.T
    return softmax(logits)   # (B, vocabSize)


def tokenize_words(text):
    if LOWERCASE_ONLY:
        return re.sub(r"[^a-z\s]", "", text.lower()).split()
    return bpe.clean_words(text)


def encode(text):
    return bpe.encode_words(tokenize_words(text), merges_rank, token_to_id)


def pad_context(ids):
    ids = ids[-seq_len:]
    if len(ids) < seq_len:
        ids = [PAD_ID] * (seq_len - len(ids)) + ids
    return ids


def predict_next(text, top_k=5):
    ids = encode(text)
    ctx = pad_context(ids)
    probs = forward([ctx])[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    print(f"\nContext: {text}")
    print(f"Top-{top_k} subword predictions:")
    for idx in top_indices:
        print(f"  '{id_to_token[idx]}'  p={probs[idx]:.4f}")


def nucleus_sample(probs, p=0.9):
    # top-p: sample from the smallest set of tokens whose probs add up to p.
    # greedy kept looping on the same few words
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumulative, p)) + 1
    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = sorted_probs[:cutoff]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    return int(np.random.choice(nucleus_idx, p=nucleus_probs))


def generate(text, num_tokens=30, mode="nucleus", top_p=0.9, top_k=1,
             repetition_penalty=1.3, penalty_window=8):
    # mode: 'nucleus', 'greedy' or 'top_k'
    ids = encode(text)
    for _ in range(num_tokens):
        ctx = pad_context(ids)
        probs = forward([ctx])[0].copy()

        recent = ids[-penalty_window:]
        if recent:
            counts = {}
            for tok in recent:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, n in counts.items():
                if tok < len(probs):
                    probs[tok] /= (repetition_penalty ** n)
            probs = probs / probs.sum()

        if mode == "greedy":
            next_id = int(np.argmax(probs))
        elif mode == "top_k":
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_probs = probs[top_indices]
            top_probs = top_probs / top_probs.sum()
            next_id = int(np.random.choice(top_indices, p=top_probs))
        else:
            next_id = nucleus_sample(probs, p=top_p)
        ids.append(next_id)

    words = bpe.decode_ids(ids, id_to_token)
    print("\nGenerated:")
    print(bpe.join_words(words))


def random_test(num_examples=10):
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = encode(text)

    sequences = []
    for i in range(len(tokens) - seq_len):
        sequences.append((tokens[i:i + seq_len], tokens[i + seq_len]))

    import random
    hits = 0
    for _ in range(num_examples):
        context, target = random.choice(sequences)
        probs = forward([context])[0]
        top5 = np.argsort(probs)[::-1][:5]
        if target in top5:
            hits += 1
        print("\nContext:", "".join(id_to_token[t] for t in context).replace(bpe.END, " "))
        print("Actual :", id_to_token[target].replace(bpe.END, ""))
        print("Top 5  :", [id_to_token[i].replace(bpe.END, "") for i in top5])

    print(f"\nTop-5 Hit Rate: {hits}/{num_examples}")


# ==========================
# Menu
# ==========================

if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("1 - Predict next token")
        print("2 - Generate text (autoregressive)")
        print("3 - Random evaluation (top-5 hit rate)")
        print("4 - Quit")
        print("=" * 50)

        choice = input("Choice: ").strip()

        if choice == "1":
            predict_next(input("\nEnter context: "))
        elif choice == "2":
            generate(input("\nEnter starting context: "))
        elif choice == "3":
            random_test(10)
        elif choice == "4":
            break
