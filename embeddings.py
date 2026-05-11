import numpy as np
import re
import random

with open('C:/Users/vinay/OneDrive/Desktop/miniLLM/corpus.txt', 'r', encoding='utf-8') as file:
    text = file.read()

text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)

words = text.split()
print(f"Number of words: {len(words)}")

print("Script continues...")  # Add this line

unique_words = sorted(set(words))
vocab_size = len(unique_words)
print(f"Number of unique words: {len(unique_words)}")

word_to_idx = {w: i for i, w in enumerate(unique_words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

tokens = [word_to_idx[w] for w in words]
print(f"First 10 tokens: {tokens[:10]}")

embedding_dim = 32 # Thumbrule: 4th root of vocabulary size i.e. = 10 but to capture more context we use 32
np.random.seed(42)

input_embeddings = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim)) / embedding_dim
output_embeddings = np.zeros((vocab_size, embedding_dim))

print(f"Embedding table shape: {input_embeddings.shape}")

word = "crime"
idx = word_to_idx[word]
embedding = input_embeddings[idx]

print(f"embedding for '{word}': {embedding}")


# Step 1: Creating context and center word pairs


max_window_size = 2 # Choosing random window so that it 
pairs = []

print("Creating training pairs...")

for i, center_word_idx in enumerate(tokens):
    window_size = random.randint(1, max_window_size)
    for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
        if i == j:
            continue
        context_word_idx = tokens[j]
        pairs.append((center_word_idx, context_word_idx))

print(f"total training pairs: {len(pairs)}")
print(f"Example pairs (word form):")
for center, context in pairs[:5]:
    print(f"  ('{idx_to_word[center]}', '{idx_to_word[context]}')")


# Step 2: Negative sampling
# for every real pair we create 5 fake pairs which teaches the model what does not belong together


negative_samples = 5

word_freq = np.zeros(vocab_size)
for token in tokens:
    word_freq[token] += 1
word_freq = word_freq ** 0.75
word_freq = word_freq / word_freq.sum()

def get_negative_samples(context_idx, num_samples):
    negatives = []
    while len(negatives) < num_samples:
        random_idx = np.random.choice(vocab_size, p=word_freq)
        if random_idx != context_idx:
            negatives.append(random_idx)
    return negatives

center, context = pairs[0]
negatives = get_negative_samples(context, negative_samples)

print(f"\nCenter word: '{idx_to_word[center]}'")
print(f"Real context: '{idx_to_word[context]}'")
print(f"Fake contexts: {[idx_to_word[n] for n in negatives]}")


# Step 3: Training Loop
# Sigmoid function: converts dot product to probability between 0 and 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.005
epochs = 20
vocab_size = len(unique_words)

print("\ntraining...")

for epoch in range(epochs):
    total_loss = 0
    random.shuffle(pairs)

    for i, (center_idx, context_idx) in enumerate(pairs):
        
        # Store original vectors BEFORE any updates
        center_vec = input_embeddings[center_idx].copy()
        context_vec = output_embeddings[context_idx].copy()

        # Forward pass: real pair should score high
        score = np.dot(center_vec, context_vec)
        prob = sigmoid(score)
        total_loss += -np.log(prob + 1e-10)

        # Update real pair
        grad = prob - 1
        input_embeddings[center_idx] -= learning_rate * grad * context_vec
        output_embeddings[context_idx] -= learning_rate * grad * center_vec

        # Negative samples: fake pairs should score low
        negatives = get_negative_samples(context_idx, negative_samples)
        for neg_idx in negatives:
            neg_vec = output_embeddings[neg_idx].copy()
            neg_score = np.dot(center_vec, neg_vec)
            neg_prob = sigmoid(neg_score)
            total_loss += -np.log(1 - neg_prob + 1e-10)

            neg_grad = neg_prob
            input_embeddings[center_idx] -= learning_rate * neg_grad * neg_vec
            output_embeddings[neg_idx] -= learning_rate * neg_grad * center_vec

        if i % 10000 == 0:
            print(f"  Epoch {epoch+1}, pair {i}/{len(pairs)}, loss: {total_loss/(i+1):.4f}")

    print(f"Epoch {epoch+1} complete. Avg loss: {total_loss/len(pairs):.4f}")

print("\nTraining complete.")

np.save("trained_embeddings.npy", input_embeddings)
np.save("word_to_idx.npy", word_to_idx)
print("Embeddings saved.")