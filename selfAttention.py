import numpy as np
import random
import re

input_embeddings = np.load("trained_embeddings.npy")
word_to_idx = np.load("word_to_idx.npy", allow_pickle=True).item()
idx_to_word = {i: w for w, i in word_to_idx.items()}

vocabSize = input_embeddings.shape[0]
embeddingDim = input_embeddings.shape[1]
d_k = embeddingDim # (32,)

print(f"Vocab size = {vocabSize}")
print(f"embedding dimentions = {embeddingDim}")

with open('C:/Users/vinay/OneDrive/Desktop/miniLLM/corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)
words = text.split()

tokens = [word_to_idx[w] for w in words if w in word_to_idx]

print(f"Number of tokens = {len(tokens)}")

# Next word prediction seq by defining input and targets

seq_len = 8
sequences = []
for i in range(len(tokens) - seq_len):
    context = tokens[i : i+ seq_len]
    target = tokens[i+ seq_len]
    sequences.append((context, target))

print(f"Total sequences = {len(sequences)}")

# Functions

def softmax(x):
    e = np.exp(x - np.max(x, axis = -1, keepdims = True))
    return e/e.sum(axis = -1, keepdims = True)

def softmax_vectors(x):
    e = np.exp(x- np.max(x))
    return e/e.sum()

# initialize parameters

np.random.seed(0)
scale = 0.01

W_Q = np.random.randn(embeddingDim, d_k) * scale # (32, 32)
W_K = np.random.randn(embeddingDim, d_k) * scale # (32, 32)
W_V = np.random.randn(embeddingDim, d_k) * scale # (32, 32)
W_O = np.random.randn(d_k, vocabSize) * scale # (32, vocab size)

# Forward

def forward(contextToken):
    X = input_embeddings[contextToken]
# each word is now projected into 3 diff vectors i.e. Query, Key, Value.
    Q = X @ W_Q # (4, 32) @ (32, 32) = (4, 32) (Query vector)
    K = X @ W_K #      " (Key vector)
    V = X @ W_V #      " (Value vector)
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.triu(np.ones_like(scores), k=1)
    scores = np.where(mask, -1e9, scores)  # K.T is just the transpose of K (4, 32)@(32, 4) = (4,4)
# it makes the matrix of perfect size of no of context tokens x no of context tokens and Cell [i, j] = dot product of Q_i and K_j = "how relevant is word_j to word_i?"
    attn_w = softmax(scores) # (4,4)
    context = attn_w @ V # Value of each words to the embeddings i.e. [4,4] @ [4,32] = [4,32]
    last_context = context[-1] # we need only the last context from the context vector to predict the next word (32,)
    logits = last_context @ W_O # logits are basically the product of context representaton against each word's learned output weight vector (32,) @ (32, vocab size) = (vocab size,)
    probs = softmax_vectors(logits) # it gives most probable next word.

    cache = dict(X = X, Q = Q, K = K, V = V,
    scores = scores, attn_w = attn_w, context = context, last_context = last_context, logits = logits, probs = probs)
    return probs, cache

# Backward
# there is a simple formula used for backward propogation i.e. Y = A @ B
# then dY/dA = dY/dB * B.T and dY/dB = dY/dA * A.T
# following that rule we back propogate each steps from the forward pass and correct each weight increasing the loss function.
# this is the chain rule of differentiation.

def backward(cache, target_idx, lr, ctx_tokens):
    global W_Q, W_K, W_V, W_O
    probs = cache['probs']
    logits = cache['logits']
    last_context = cache['last_context']
    context = cache['context']
    attn_w = cache['attn_w']
    scores = cache['scores']
    V = cache['V']
    K = cache['K']
    Q = cache['Q']
    X = cache['X']

# Starting from right to left as backpropogation is just the opposite of forward pass
    d_logits = probs.copy()  # d_logits is just the derivative of logits w.r.t. target_index which basically how wrong the predicted vector is
    d_logits[target_idx] -= 1.0

    last_ctx = context[-1]

    dW_O = np.outer(last_ctx, d_logits)

    d_last_ctx = W_O @ d_logits

    d_context = np.zeros_like(context)
    d_context[-1] = d_last_ctx

    d_attn_w = d_context @ V.T
    d_V = attn_w.T @ d_context

    d_scores = np.zeros_like(attn_w)
    for i in range(seq_len):
        a = attn_w[i]
        jac = np.diag(a) - np.outer(a,a)
        d_scores[i] = jac @ d_attn_w[i]

    d_scores /= np.sqrt(d_k)

    d_Q = d_scores @ K
    d_K = d_scores.T @ Q

    dW_Q = X.T @ d_Q
    dW_K = X.T @ d_K
    dW_V = X.T @ d_V

    d_X = d_Q @ W_Q.T + d_K @ W_K.T + d_V @ W_V.T
    W_Q -= lr * dW_Q
    W_K -= lr * dW_K
    W_V -= lr * dW_V
    W_O -= lr * dW_O

    # train embeddings too
    global input_embeddings
    for j, tok_idx in enumerate(ctx_tokens):
        input_embeddings[int(tok_idx)] -= lr * d_X[j]


# Training loop
Epochs = 15
try:
    for epoch in range(Epochs):
        random.shuffle(sequences)
        total_loss = 0.0

        for i, (context, target_idx) in enumerate(sequences):
            lr = 0.01 / (1 + 0.0001 * i)
            probs, cache = forward(context)
            loss = -np.log(probs[target_idx] + 1e-10)
            total_loss += loss
            backward(cache, target_idx, lr, context)

            if i % 20_000 == 0:
                print(f"Epoch {epoch+1} | step {i:>6}/{len(sequences)} | "
                      f"loss {total_loss / (i + 1):.4f}")

        avg_loss = total_loss / len(sequences)
        print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

except KeyboardInterrupt:
    print("\nTraining interrupted — saving weights...")

finally:
    np.save("W_Q.npy", W_Q)
    np.save("W_K.npy", W_K)
    np.save("W_V.npy", W_V)
    np.save("W_O.npy", W_O)
    print("Weights saved.")

def predict_next(words_in, top_k=5):
    ctx = [word_to_idx[w] for w in words_in if w in word_to_idx]
    ctx = ctx[-seq_len:]          # take last SEQ_LEN tokens
    if len(ctx) < seq_len:
        ctx = [0] * (seq_len - len(ctx)) + ctx   # pad with index 0
 
    probs, _ = forward(ctx)
    top_indices = np.argsort(probs)[::-1][:top_k]
    print(f"\nContext : {words_in}")
    print(f"Top-{top_k} predictions:")
    for idx in top_indices:
        print(f"  '{idx_to_word[idx]}'  p={probs[idx]:.4f}")