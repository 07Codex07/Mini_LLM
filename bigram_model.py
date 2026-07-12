import numpy as np

# ── 1. Data ──────────────────────────────────────────────────────────────────
text = "in the midst of chaos, there is also opportunity. the journey of learning is never easy, but it is always rewarding. every mistake you make is a step closer to understanding. intelligence is not just about knowing facts, but about connecting ideas and seeing patterns where others see noise. persistence and curiosity are the two most powerful tools you can carry. over time, even the most complex problems begin to feel manageable, and what once seemed impossible becomes second nature."

# ── 2. Tokenizer ─────────────────────────────────────────────────────────────
chars = sorted(set(text))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}  # cleaner reverse

print(f"Vocab size: {vocab_size} | Characters: {''.join(chars)}")

# ── 3. Tokenize the text ─────────────────────────────────────────────────────
tokens = [char_to_idx[ch] for ch in text]

# ── 4. Build bigram count matrix ─────────────────────────────────────────────
bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)

for i in range(len(tokens) - 1):
    bigram_counts[tokens[i], tokens[i + 1]] += 1

# ── 5. Softmax ───────────────────────────────────────────────────────────────
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ── 6. Normalize rows → probability matrix ───────────────────────────────────
bigram_probs = np.zeros_like(bigram_counts)
for i in range(vocab_size):
    row = bigram_counts[i]
    if row.sum() > 0:
        bigram_probs[i] = softmax(row)

# ── 7. Text generation ───────────────────────────────────────────────────────
def generate(start_char, max_length=100, temperature=1.0):
    """
    temperature=1.0  → normal sampling
    temperature=0.5  → more focused / repetitive  
    temperature=2.0  → more random / creative
    """
    if start_char not in char_to_idx:
        raise ValueError(f"'{start_char}' not in vocabulary")

    current_idx = char_to_idx[start_char]
    result = [start_char]

    for _ in range(max_length - 1):
        probs = bigram_probs[current_idx]

        # Guard: if char never seen as prefix, stop
        if probs.sum() == 0:
            break

        # Apply temperature scaling
        log_probs = np.log(probs + 1e-10) / temperature
        scaled = np.exp(log_probs - np.max(log_probs))
        scaled /= scaled.sum()

        next_idx = np.random.choice(vocab_size, p=scaled)
        result.append(idx_to_char[next_idx])
        current_idx = next_idx

    return ''.join(result)

# ── 8. Try it ────────────────────────────────────────────────────────────────
print("\n--- temperature 0.5 (focused) ---")
print(generate("t", 120, temperature=0.5))

print("\n--- temperature 1.0 (normal) ---")
print(generate("t", 120, temperature=1.0))

print("\n--- temperature 2.0 (chaotic) ---")
print(generate("t", 120, temperature=2.0))