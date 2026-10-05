import math
import numpy as np
import os
import pickle

import bpe

# ==========================
# Config
# ==========================

CORPUS_FILE = "corpus_v2_clean.txt"
embeddingsDim = 192  # was 128, corpus is ~8x bigger now
num_heads = 4
d_head = embeddingsDim // num_heads
seq_len = 64         # was 32. last position (full context) still did better than the average, so more context helps
d_ff = 4 * embeddingsDim
num_layers = 3

NUM_MERGES = 6000
Epochs = 15          # 1 epoch = roughly one pass over the training tokens
BATCH_SIZE = 256     # throughput peaked around 256-512 when I benchmarked it
clip_norm = 5.0
DROPOUT_RATE = 0.1
VAL_FRACTION = 0.05

# float32 everywhere. keep scalars as python floats - with numpy 2 a np.float64 scalar
# (like np.sqrt(192)) quietly turns float32 arrays back into float64
DTYPE = np.float32

EMBED_SCALE = math.sqrt(embeddingsDim)

# ==========================
# Tokenizer: BPE trained fresh on this corpus
# ==========================

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

word_freqs = bpe.get_word_freqs(raw_text)
print(f"Training BPE ({NUM_MERGES} merges) over {len(word_freqs)} unique words...")
merges, vocab_symbols = bpe.train_bpe(word_freqs, NUM_MERGES, verbose=False)
merges_rank = bpe.build_merges_rank(merges)
token_to_id, id_to_token = bpe.build_vocab_index(vocab_symbols)

vocabSize = len(token_to_id)
PAD_ID = vocabSize

with open("bpe_data.pkl", "wb") as f:
    pickle.dump(dict(merges=merges, token_to_id=token_to_id, id_to_token=id_to_token), f)

print(f"BPE vocab size: {vocabSize}")

words = bpe.clean_words(raw_text)
# encode each unique word once instead of every occurrence
bpe_cache = {w: bpe.encode_words([w], merges_rank, token_to_id) for w in word_freqs}
tokens_arr = np.fromiter((t for w in words for t in bpe_cache[w]), dtype=np.int32)
print(f"Corpus is now {len(tokens_arr)} subword tokens")

# last 5% of the corpus is validation, never trained on.
# (used to split random windows instead, but windows overlap by seq_len-1 tokens so val leaked into train)
n_val_tokens = int(len(tokens_arr) * VAL_FRACTION)
train_tokens = tokens_arr[:-n_val_tokens]
val_tokens = tokens_arr[-n_val_tokens:]

# window of seq_len+1: input = window[:-1], target = window[1:]
# training windows get random start positions each epoch
WINDOW_OFFSETS = np.arange(seq_len + 1)
num_batches = (len(train_tokens) // seq_len) // BATCH_SIZE


def gather_windows(token_array, starts):
    w = token_array[starts[:, None] + WINDOW_OFFSETS]   # (B, seq_len + 1)
    return w[:, :-1], w[:, 1:]


# val windows don't overlap
val_starts = np.arange(0, len(val_tokens) - seq_len, seq_len)
num_val_batches = len(val_starts) // BATCH_SIZE
val_contexts, val_targets = gather_windows(val_tokens, val_starts[:num_val_batches * BATCH_SIZE])

print(f"Training tokens: {len(train_tokens)} -> {num_batches} batches of {BATCH_SIZE} per epoch (seq_len={seq_len})")
print(f"Validation tokens held out: {len(val_tokens)} -> {num_val_batches} batches (never trained on)")

# ==========================
# Init
# ==========================

np.random.seed(0)

# scale down the layers that write into the residual stream (like GPT-2), otherwise it grows with depth
RESID_SCALE = 1.0 / math.sqrt(2 * num_layers)


def xavier(fan_in, fan_out, scale=1.0):
    return (np.random.randn(fan_in, fan_out) * math.sqrt(2.0 / fan_in) * scale).astype(DTYPE)


input_embeddings = (np.random.randn(vocabSize + 1, embeddingsDim) / math.sqrt(embeddingsDim)).astype(DTYPE)


def build_positional_encoding(seq_len, dim):
    matrix = np.zeros((seq_len, dim))
    for pos in range(seq_len):
        for i in range(dim):
            angle = pos / (10000 ** ((2 * (i // 2)) / dim))
            matrix[pos, i] = np.sin(angle) if i % 2 == 0 else np.cos(angle)
    return matrix


pos_encoding = build_positional_encoding(seq_len, embeddingsDim)[None, :, :].astype(DTYPE)  # (1, seq_len, dim)

layers = []
for _ in range(num_layers):
    layers.append(dict(
        W_Q=xavier(embeddingsDim, embeddingsDim),
        W_K=xavier(embeddingsDim, embeddingsDim),
        W_V=xavier(embeddingsDim, embeddingsDim),
        W_O_attn=xavier(embeddingsDim, embeddingsDim, scale=RESID_SCALE),
        gamma1=np.ones(embeddingsDim, dtype=DTYPE), beta1=np.zeros(embeddingsDim, dtype=DTYPE),
        W_ff1=xavier(embeddingsDim, d_ff),
        W_ff2=xavier(d_ff, embeddingsDim, scale=RESID_SCALE),
        gamma2=np.ones(embeddingsDim, dtype=DTYPE), beta2=np.zeros(embeddingsDim, dtype=DTYPE),
    ))

# pre-LN needs one more layernorm at the end before the output layer
final_ln = dict(gamma_f=np.ones(embeddingsDim, dtype=DTYPE), beta_f=np.zeros(embeddingsDim, dtype=DTYPE))

# Adam state
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
WEIGHT_DECAY = 0.01  # AdamW style, not applied to layernorm params
_NO_DECAY_KEYS = {'gamma1', 'beta1', 'gamma2', 'beta2', 'gamma_f', 'beta_f'}

layer_opt_state = [
    {k: dict(m=np.zeros_like(v), v=np.zeros_like(v)) for k, v in layer.items()}
    for layer in layers
]
final_ln_opt_state = {k: dict(m=np.zeros_like(v), v=np.zeros_like(v)) for k, v in final_ln.items()}
embed_opt_state = dict(m=np.zeros_like(input_embeddings), v=np.zeros_like(input_embeddings))

TRAINING_STATE_FILE = "training_state.pkl"


def save_training_state(global_step, next_epoch, best_val_loss):
    # save adam's m and v too, otherwise resuming resets the optimizer
    with open(TRAINING_STATE_FILE, "wb") as f:
        pickle.dump(dict(
            global_step=global_step,
            next_epoch=next_epoch,
            best_val_loss=best_val_loss,
            layer_opt_state=layer_opt_state,
            final_ln_opt_state=final_ln_opt_state,
            embed_opt_state=embed_opt_state,
        ), f)


def try_resume():
    # final_ln file only exists for pre-LN checkpoints, so old post-LN ones won't get loaded by mistake
    needed = ["transformer_embeddings.npy", "transformer_final_ln.npz", TRAINING_STATE_FILE]
    if not all(os.path.exists(p) for p in needed):
        return None

    input_embeddings[:] = np.load("transformer_embeddings.npy")
    for i, layer in enumerate(layers):
        loaded = np.load(f"transformer_layer{i}.npz")
        for key in layer:
            layer[key][:] = loaded[key]
    loaded = np.load("transformer_final_ln.npz")
    for key in final_ln:
        final_ln[key][:] = loaded[key]

    with open(TRAINING_STATE_FILE, "rb") as f:
        state = pickle.load(f)
    for saved, current in zip(state['layer_opt_state'] + [state['final_ln_opt_state']],
                              layer_opt_state + [final_ln_opt_state]):
        for key in current:
            current[key]['m'][:] = saved[key]['m']
            current[key]['v'][:] = saved[key]['v']
    embed_opt_state['m'][:] = state['embed_opt_state']['m']
    embed_opt_state['v'][:] = state['embed_opt_state']['v']

    return state['global_step'], state['next_epoch'], state['best_val_loss']


def adam_update(param, grad, state, lr, t, weight_decay=0.0):
    state['m'][:] = ADAM_B1 * state['m'] + (1 - ADAM_B1) * grad
    state['v'][:] = ADAM_B2 * state['v'] + (1 - ADAM_B2) * (grad ** 2)
    m_hat = state['m'] / (1 - ADAM_B1 ** t)
    v_hat = state['v'] / (1 - ADAM_B2 ** t)
    param -= lr * (m_hat / (np.sqrt(v_hat) + ADAM_EPS) + weight_decay * param)


causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

# ==========================
# Functions (everything is batched, shapes are (B, ...))
# ==========================


def mm(X, W):
    # (B, S, n) @ (n, m) as one 2D matmul - numpy does the 3D version as B small matmuls, 2-3x slower
    return (X.reshape(-1, X.shape[-1]) @ W).reshape(*X.shape[:-1], W.shape[-1])


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def target_probs(probs, targets):
    # prob of the actual next token at each position
    B, S = targets.shape
    return probs[np.arange(B)[:, None], np.arange(S)[None, :], targets]


def dropout_forward(x, rate, training):
    if not training or rate == 0.0:
        return x, None
    mask = (np.random.rand(*x.shape) > rate).astype(x.dtype) / x.dtype.type(1.0 - rate)  # inverted dropout
    return x * mask, mask


def dropout_backward(dout, mask):
    if mask is None:
        return dout
    return dout * mask


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    return out, (x_norm, var, eps)


def layernorm_backward(dout, gamma, cache):
    x_norm, var, eps = cache
    D = dout.shape[-1]
    std_inv = 1.0 / np.sqrt(var + eps)

    dxnorm = dout * gamma
    dx = std_inv * (
        dxnorm
        - dxnorm.mean(axis=-1, keepdims=True)
        - x_norm * np.sum(dxnorm * x_norm, axis=-1, keepdims=True) / D
    )

    reduce_axes = tuple(range(dout.ndim - 1))  # sum over everything except the feature dim
    dgamma = np.sum(dout * x_norm, axis=reduce_axes)
    dbeta = np.sum(dout, axis=reduce_axes)
    return dx, dgamma, dbeta


def split_heads(t, B):
    return t.reshape(B, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)


def merge_heads(t, B):
    return t.transpose(0, 2, 1, 3).reshape(B, seq_len, embeddingsDim)


def attention_forward(X, layer):
    B = X.shape[0]
    Q, K, V = mm(X, layer['W_Q']), mm(X, layer['W_K']), mm(X, layer['W_V'])
    Qh, Kh, Vh = split_heads(Q, B), split_heads(K, B), split_heads(V, B)

    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / math.sqrt(d_head)
    scores = np.where(causal_mask, -1e9, scores)
    attn_w = softmax(scores)
    context_h = attn_w @ Vh
    context = merge_heads(context_h, B)
    attn_out = mm(context, layer['W_O_attn'])

    cache = dict(X=X, Qh=Qh, Kh=Kh, Vh=Vh, attn_w=attn_w, context=context)
    return attn_out, cache


def attention_backward(d_attn_out, cache, layer):
    X, Qh, Kh, Vh = cache['X'], cache['Qh'], cache['Kh'], cache['Vh']
    attn_w, context = cache['attn_w'], cache['context']
    B = X.shape[0]

    dW_O_attn = context.reshape(-1, embeddingsDim).T @ d_attn_out.reshape(-1, embeddingsDim)
    d_context = mm(d_attn_out, layer['W_O_attn'].T)
    d_context_h = split_heads(d_context, B)

    d_attn_w = d_context_h @ Vh.transpose(0, 1, 3, 2)
    d_Vh = attn_w.transpose(0, 1, 3, 2) @ d_context_h

    # softmax backward without building the jacobian: dz = p * (dp - sum(dp * p))
    d_scores = attn_w * (d_attn_w - np.sum(d_attn_w * attn_w, axis=-1, keepdims=True))
    d_scores /= math.sqrt(d_head)

    d_Qh = d_scores @ Kh
    d_Kh = d_scores.transpose(0, 1, 3, 2) @ Qh

    d_Q, d_K, d_V = merge_heads(d_Qh, B), merge_heads(d_Kh, B), merge_heads(d_Vh, B)

    dW_Q = X.reshape(-1, embeddingsDim).T @ d_Q.reshape(-1, embeddingsDim)
    dW_K = X.reshape(-1, embeddingsDim).T @ d_K.reshape(-1, embeddingsDim)
    dW_V = X.reshape(-1, embeddingsDim).T @ d_V.reshape(-1, embeddingsDim)

    d_X = mm(d_Q, layer['W_Q'].T) + mm(d_K, layer['W_K'].T) + mm(d_V, layer['W_V'].T)
    grads = dict(W_Q=dW_Q, W_K=dW_K, W_V=dW_V, W_O_attn=dW_O_attn)
    return d_X, grads


GELU_C = math.sqrt(2.0 / math.pi)


def gelu_forward(x):
    # tanh approximation (same one GPT-2 uses). x*x*x because x**3 is way slower in numpy
    t = np.tanh(GELU_C * (x + 0.044715 * (x * x * x)))
    return 0.5 * x * (1.0 + t), t


def gelu_backward(d_out, x, t):
    # u = c(x + 0.044715x^3), gelu = 0.5x(1 + tanh(u))
    # d/dx = 0.5(1 + t) + 0.5x(1 - t^2) * du/dx
    du_dx = GELU_C * (1.0 + 3 * 0.044715 * (x * x))
    return d_out * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx)


def ffn_forward(X, layer):
    pre = mm(X, layer['W_ff1'])
    h, t = gelu_forward(pre)
    out = mm(h, layer['W_ff2'])
    return out, dict(X=X, pre=pre, h=h, t=t)


def ffn_backward(d_out, cache, layer):
    X, pre, h, t = cache['X'], cache['pre'], cache['h'], cache['t']
    dW_ff2 = h.reshape(-1, d_ff).T @ d_out.reshape(-1, embeddingsDim)
    d_h = mm(d_out, layer['W_ff2'].T)
    d_pre = gelu_backward(d_h, pre, t)
    dW_ff1 = X.reshape(-1, embeddingsDim).T @ d_pre.reshape(-1, d_ff)
    d_X = mm(d_pre, layer['W_ff1'].T)
    return d_X, dict(W_ff1=dW_ff1, W_ff2=dW_ff2)


def block_forward(X, layer, training=True):
    # pre-LN: x + f(LN(x)) instead of LN(x + f(x)).
    # residual path never goes through a layernorm, trains more stable
    a_in, ln1_cache = layernorm_forward(X, layer['gamma1'], layer['beta1'])
    attn_out, attn_cache = attention_forward(a_in, layer)
    attn_out, drop1_mask = dropout_forward(attn_out, DROPOUT_RATE, training)
    X1 = X + attn_out

    f_in, ln2_cache = layernorm_forward(X1, layer['gamma2'], layer['beta2'])
    ff_out, ff_cache = ffn_forward(f_in, layer)
    ff_out, drop2_mask = dropout_forward(ff_out, DROPOUT_RATE, training)
    X2 = X1 + ff_out

    cache = dict(attn_cache=attn_cache, ln1_cache=ln1_cache, ff_cache=ff_cache, ln2_cache=ln2_cache,
                 drop1_mask=drop1_mask, drop2_mask=drop2_mask)
    return X2, cache


def block_backward(dX2, cache, layer):
    d_ff_out = dropout_backward(dX2, cache['drop2_mask'])
    d_f_in, ffn_grads = ffn_backward(d_ff_out, cache['ff_cache'], layer)
    d_X1_from_ffn, dgamma2, dbeta2 = layernorm_backward(d_f_in, layer['gamma2'], cache['ln2_cache'])
    dX1 = dX2 + d_X1_from_ffn                                    # residual

    d_attn_out = dropout_backward(dX1, cache['drop1_mask'])
    d_a_in, attn_grads = attention_backward(d_attn_out, cache['attn_cache'], layer)
    d_X_from_attn, dgamma1, dbeta1 = layernorm_backward(d_a_in, layer['gamma1'], cache['ln1_cache'])
    dX = dX1 + d_X_from_attn                                     # residual

    grads = dict(attn_grads, **ffn_grads, gamma1=dgamma1, beta1=dbeta1, gamma2=dgamma2, beta2=dbeta2)
    return dX, grads


def forward(context_ids_batch, training=True):
    idx = np.asarray(context_ids_batch)        # (B, seq_len)
    X = input_embeddings[idx] * EMBED_SCALE + pos_encoding

    caches = []
    for layer in layers:
        X, cache = block_forward(X, layer, training=training)
        caches.append(cache)

    X_final, lnf_cache = layernorm_forward(X, final_ln['gamma_f'], final_ln['beta_f'])

    # predict at every position, not just the last one
    E_real = input_embeddings[:vocabSize]
    logits = mm(X_final, E_real.T)              # (B, seq_len, vocabSize)
    probs = softmax(logits)

    return probs, dict(idx=idx, caches=caches, X_final=X_final, lnf_cache=lnf_cache)


def compute_grads(probs, cache, target_batch):
    # note: overwrites probs (reused as d_logits, it's the biggest array here)
    idx = cache['idx']                          # (B, seq_len)
    X_final = cache['X_final']                  # (B, seq_len, dim)
    B, S = idx.shape

    d_logits = probs
    d_logits[np.arange(B)[:, None], np.arange(S)[None, :], target_batch] -= 1.0
    d_logits /= (B * S)  # mean over all positions

    E_real = input_embeddings[:vocabSize]
    d_embed_from_head = d_logits.reshape(-1, vocabSize).T @ X_final.reshape(-1, embeddingsDim)
    dX_final = mm(d_logits, E_real)             # (B, seq_len, dim)
    dX, dgamma_f, dbeta_f = layernorm_backward(dX_final, final_ln['gamma_f'], cache['lnf_cache'])

    all_grads = []
    for layer, blk_cache in zip(reversed(layers), reversed(cache['caches'])):
        dX, grads = block_backward(dX, blk_cache, layer)
        all_grads.append(grads)
    all_grads.reverse()

    d_embed_total = np.zeros_like(input_embeddings)
    d_embed_total[:vocabSize] += d_embed_from_head
    idx_flat = idx.reshape(-1)
    dX_flat = (dX * EMBED_SCALE).reshape(-1, embeddingsDim)
    np.add.at(d_embed_total, idx_flat, dX_flat)   # add.at so repeated tokens add up properly

    return all_grads, dict(gamma_f=dgamma_f, beta_f=dbeta_f), d_embed_total


def global_grad_norm(all_grads, final_grads, d_embed_total):
    total_sq = sum(float(np.sum(g ** 2)) for grads in all_grads + [final_grads] for g in grads.values())
    total_sq += float(np.sum(d_embed_total ** 2))
    return math.sqrt(total_sq)


def backward(probs, cache, target_batch, lr, t):
    all_grads, final_grads, d_embed_total = compute_grads(probs, cache, target_batch)

    # clip by one global norm across all grads (per-tensor clipping changed the step direction)
    total_norm = global_grad_norm(all_grads, final_grads, d_embed_total)
    scale = clip_norm / total_norm if total_norm > clip_norm else 1.0

    for params, grads, opt_state in zip(layers + [final_ln], all_grads + [final_grads],
                                        layer_opt_state + [final_ln_opt_state]):
        for key in grads:
            wd = 0.0 if key in _NO_DECAY_KEYS else WEIGHT_DECAY
            adam_update(params[key], grads[key] * scale, opt_state[key], lr, t, weight_decay=wd)

    adam_update(input_embeddings, d_embed_total * scale, embed_opt_state, lr, t, weight_decay=WEIGHT_DECAY)


def evaluate():
    # val loss, dropout off
    total = 0.0
    for b in range(num_val_batches):
        batch_contexts = val_contexts[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        batch_targets = val_targets[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        probs, _ = forward(batch_contexts, training=False)
        total += -np.mean(np.log(target_probs(probs, batch_targets) + 1e-10))
    return total / num_val_batches


def snapshot_weights():
    return dict(
        input_embeddings=input_embeddings.copy(),
        layers=[{k: v.copy() for k, v in layer.items()} for layer in layers],
        final_ln={k: v.copy() for k, v in final_ln.items()},
    )


def save_weights(embeddings, layers_to_save, final_ln_to_save, suffix=""):
    np.save(f"transformer_embeddings{suffix}.npy", embeddings)
    for i, layer in enumerate(layers_to_save):
        np.savez(f"transformer_layer{i}{suffix}.npz", **layer)
    np.savez(f"transformer_final_ln{suffix}.npz", **final_ln_to_save)


def save_config():
    with open("model_config.pkl", "wb") as f:
        pickle.dump(dict(seq_len=seq_len, embeddingsDim=embeddingsDim, num_heads=num_heads,
                         num_layers=num_layers, lowercase_only=False, corpus_file=CORPUS_FILE,
                         val_fraction=VAL_FRACTION, pre_ln=True, activation="gelu"), f)


# ==========================
# Training loop
# ==========================

base_lr = 1e-3  # post-LN version used 3e-4, pre-LN handles higher
min_lr = base_lr * 0.1
total_steps = Epochs * num_batches
WARMUP_STEPS = min(300, max(10, total_steps // 10))
print(f"Total training steps (batches): {total_steps} (warmup: {WARMUP_STEPS} steps)")


def get_lr(step):
    # linear warmup, then cosine decay down to min_lr
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    progress = min(1.0, (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


best_val_loss = float("inf")
best_snapshot = None
global_step = 0
start_epoch = 0

resumed = try_resume()
if resumed is not None:
    global_step, start_epoch, best_val_loss = resumed
    print(f"Resumed from checkpoint: global_step={global_step}, "
          f"starting at epoch {start_epoch + 1}, best_val_loss={best_val_loss:.4f}")
    if start_epoch >= Epochs:
        print(f"Checkpoint already completed all {Epochs} epochs — nothing to do.")

epoch = start_epoch - 1  # so the finally block works even if the loop never runs

try:
    for epoch in range(start_epoch, Epochs):
        epoch_starts = np.random.randint(0, len(train_tokens) - seq_len, size=num_batches * BATCH_SIZE)
        total_loss = 0.0

        for b in range(num_batches):
            batch_contexts, batch_targets = gather_windows(train_tokens, epoch_starts[b * BATCH_SIZE:(b + 1) * BATCH_SIZE])

            lr = get_lr(global_step)
            probs, cache = forward(batch_contexts)
            loss = -np.mean(np.log(target_probs(probs, batch_targets) + 1e-10))
            total_loss += loss
            backward(probs, cache, batch_targets, lr, global_step + 1)
            global_step += 1

            if b % 200 == 0:
                print(f"Epoch {epoch + 1} | batch {b:>6}/{num_batches} | "
                      f"loss {total_loss / (b + 1):.4f} | lr {lr:.5f}")

        avg_loss = total_loss / num_batches
        val_loss = evaluate()
        print(f"Epoch {epoch + 1} complete. Train loss: {avg_loss:.4f} | Validation loss: {val_loss:.4f} "
              f"(perplexity {math.exp(val_loss):.1f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_snapshot = snapshot_weights()
            print(f"  -> new best validation loss, snapshot saved")

        # save every epoch, a crash or colab disconnect never reaches the finally block
        save_weights(input_embeddings, layers, final_ln)
        save_training_state(global_step, epoch + 1, best_val_loss)
        if best_snapshot is not None:
            save_weights(best_snapshot["input_embeddings"], best_snapshot["layers"], best_snapshot["final_ln"], suffix="_best")
        save_config()

except KeyboardInterrupt:
    print("\nTraining interrupted — saving weights...")

finally:
    save_weights(input_embeddings, layers, final_ln)
    save_training_state(global_step, epoch + 1, best_val_loss)
    if best_snapshot is not None:
        save_weights(best_snapshot["input_embeddings"], best_snapshot["layers"], best_snapshot["final_ln"], suffix="_best")
        print(f"Best validation loss: {best_val_loss:.4f} (saved as transformer_*_best files)")
    save_config()
    print("Weights saved.")
