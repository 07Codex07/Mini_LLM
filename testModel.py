import numpy as np
import re
import random

# ==========================
# Load trained parameters
# ==========================

input_embeddings = np.load("trained_embeddings.npy")

word_to_idx = np.load(
    "word_to_idx.npy",
    allow_pickle=True
).item()

idx_to_word = {
    i: w for w, i in word_to_idx.items()
}

W_Q = np.load("W_Q.npy")
W_K = np.load("W_K.npy")
W_V = np.load("W_V.npy")
W_O = np.load("W_O.npy")

vocabSize = input_embeddings.shape[0]
embeddingDim = input_embeddings.shape[1]
d_k = embeddingDim
seq_len = 8

print(f"Vocabulary Size : {vocabSize}")
print(f"Embedding Dim   : {embeddingDim}")

# ==========================
# Rebuild sequences
# ==========================

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.lower()
text = re.sub(r"[^a-z\s]", "", text)

words = text.split()

tokens = [
    word_to_idx[w]
    for w in words
    if w in word_to_idx
]

sequences = []

for i in range(len(tokens) - seq_len):
    context = tokens[i:i + seq_len]
    target = tokens[i + seq_len]
    sequences.append((context, target))

print(f"Sequences loaded: {len(sequences)}")


# ==========================
# Functions
# ==========================

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softmax_vector(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def forward(context_tokens):

    X = input_embeddings[context_tokens]

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T / np.sqrt(d_k)

    mask = np.triu(
        np.ones_like(scores),
        k=1
    )

    scores = np.where(
        mask,
        -1e9,
        scores
    )

    attn_w = softmax(scores)

    context = attn_w @ V

    last_context = context[-1]

    logits = last_context @ W_O

    probs = softmax_vector(logits)

    return probs


def predict_next(words_in, top_k=10):

    ctx = [
        word_to_idx[w]
        for w in words_in
        if w in word_to_idx
    ]

    ctx = ctx[-seq_len:]

    if len(ctx) < seq_len:
        ctx = [0] * (seq_len - len(ctx)) + ctx

    probs = forward(ctx)

    top_indices = np.argsort(probs)[::-1][:top_k]

    print("\nContext:")
    print(" ".join(words_in))

    print("\nTop Predictions:\n")

    for idx in top_indices:
        print(
            f"{idx_to_word[idx]:20s}"
            f"{probs[idx]:.5f}"
        )


# ==========================
# Random Evaluation
# ==========================

def random_test(num_examples=5):

    print("\n" + "=" * 60)
    print("RANDOM TESTS")
    print("=" * 60)

    hits = 0

    for _ in range(num_examples):

        context, target = random.choice(sequences)

        probs = forward(context)

        top5 = np.argsort(probs)[::-1][:5]

        target_word = idx_to_word[target]

        if target in top5:
            hits += 1

        print("\nContext:")
        print(
            " ".join(
                idx_to_word[t]
                for t in context
            )
        )

        print(f"Actual : {target_word}")

        print("Top 5  :")

        for idx in top5:
            print(
                f"   {idx_to_word[idx]}"
            )

    print("\nTop-5 Hit Rate:")
    print(
        f"{hits}/{num_examples}"
    )


# ==========================
# Main Menu
# ==========================

while True:

    print("\n")
    print("=" * 50)
    print("1 - Predict next word")
    print("2 - Random evaluation")
    print("3 - Quit")
    print("=" * 50)

    choice = input("Choice: ").strip()

    if choice == "1":

        text = input(
            "\nEnter context: "
        ).lower()

        words = re.findall(
            r"[a-z]+",
            text
        )

        predict_next(words)

    elif choice == "2":

        random_test(10)

    elif choice == "3":

        break