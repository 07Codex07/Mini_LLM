import numpy as np
import re

# ── Load trained weights and vocab ──────────────────────────────────────────
W_Q = np.load("W_Q.npy")
W_K = np.load("W_K.npy")
W_V = np.load("W_V.npy")
W_O = np.load("W_O.npy")
input_embeddings = np.load("trained_embeddings.npy")
word_to_idx = np.load("word_to_idx.npy", allow_pickle=True).item()
idx_to_word  = {i: w for w, i in word_to_idx.items()}

embeddingDim = input_embeddings.shape[1]
d_k          = embeddingDim
seq_len      = 8

# ── Helpers ──────────────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def softmax_vec(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def forward(ctx_tokens):
    X      = input_embeddings[ctx_tokens]
    Q      = X @ W_Q
    K      = X @ W_K
    V      = X @ W_V
    scores = Q @ K.T / np.sqrt(d_k)
    mask   = np.triu(np.ones_like(scores), k=1)
    scores = np.where(mask, -1e9, scores)
    attn_w = softmax(scores)
    ctx_   = attn_w @ V
    last   = ctx_[-1]
    logits = last @ W_O
    probs  = softmax_vec(logits)
    return probs, dict(X=X, Q=Q, K=K, V=V, scores=scores,
                       attn_w=attn_w, context=ctx_, last_context=last,
                       logits=logits, probs=probs)

def get_embedding(word):
    idx = word_to_idx.get(word)
    if idx is None:
        return None
    return input_embeddings[idx].copy()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def l2_dist(a, b):
    return np.linalg.norm(a - b)

# ── Test: watch how a word's embedding drifts over N gradient steps ──────────
def track_embedding_drift(context_words, target_word, steps=50, lr=0.005):
    """
    Runs `steps` forward+backward passes on the given context->target pair
    and prints how the target word's embedding changes each step.
    """
    target_idx = word_to_idx.get(target_word)
    if target_idx is None:
        print(f"'{target_word}' not in vocab.")
        return

    # build context token list
    ctx = [word_to_idx[w] for w in context_words if w in word_to_idx]
    ctx = ctx[-seq_len:]
    if len(ctx) < seq_len:
        ctx = [0] * (seq_len - len(ctx)) + ctx

    # snapshot of every word's embedding in the context before training
    snapshots = {}
    for w in context_words:
        if w in word_to_idx:
            snapshots[w] = get_embedding(w)

    print(f"\n{'='*60}")
    print(f"Context  : {context_words}")
    print(f"Target   : '{target_word}'")
    print(f"{'='*60}")
    print(f"{'Step':>5}  {'Loss':>8}  " +
          "  ".join([f"Δ‖{w}‖" for w in snapshots]))
    print("-" * 60)

    for step in range(steps):
        emb_before = {w: get_embedding(w) for w in snapshots}

        probs, cache = forward(ctx)
        loss = -np.log(probs[target_idx] + 1e-10)

        # ── inline backward (mirrors your original, no globals needed here) ──
        d_logits             = probs.copy()
        d_logits[target_idx] -= 1.0

        dW_O      = np.outer(cache['last_context'], d_logits)
        d_last    = W_O @ d_logits
        d_context = np.zeros_like(cache['context'])
        d_context[-1] = d_last

        d_attn_w = d_context @ cache['V'].T
        d_V      = cache['attn_w'].T @ d_context

        d_scores = np.zeros_like(cache['attn_w'])
        for i in range(seq_len):
            a = cache['attn_w'][i]
            jac = np.diag(a) - np.outer(a, a)
            d_scores[i] = jac @ d_attn_w[i]
        d_scores /= np.sqrt(d_k)

        d_Q = d_scores @ cache['K']
        d_K = d_scores.T @ cache['Q']
        d_X = d_Q @ W_Q.T + d_K @ W_K.T + (cache['attn_w'].T @ d_context) @ W_V.T

        # update only embeddings (weights are frozen — we loaded trained ones)
        for j, tok_idx in enumerate(ctx):
            input_embeddings[int(tok_idx)] -= lr * d_X[j]

        # measure drift for each context word
        drifts = []
        for w in snapshots:
            emb_after = get_embedding(w)
            drifts.append(f"{l2_dist(emb_before[w], emb_after):>8.5f}")

        if step % 5 == 0 or step == steps - 1:
            print(f"{step:>5}  {loss:>8.4f}  {'  '.join(drifts)}")

    print(f"\nFinal cosine similarity of each context word's embedding")
    print(f"vs its snapshot before these {steps} steps:")
    for w in snapshots:
        final = get_embedding(w)
        sim   = cosine_sim(snapshots[w], final)
        drift = l2_dist(snapshots[w], final)
        print(f"  '{w:15s}'  cosine={sim:.4f}  L2 drift={drift:.5f}")


# ── Run it ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # swap these out for words actually in your corpus
    context = ["above", "all", "dont", "lie", "to", "yourself", "the", "man"]
    target  = "who"

    track_embedding_drift(context, target, steps=50, lr=0.005)