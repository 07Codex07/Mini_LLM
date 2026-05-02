import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

text = "in the midst of chaos, there is also opportunity. the journey of learning is never easy, but it is always rewarding. every mistake you make is a step closer to understanding. intelligence is not just about knowing facts, but about connecting ideas and seeing patterns where others see noise. persistence and curiosity are the two most powerful tools you can carry. over time, even the most complex problems begin to feel manageable, and what once seemed impossible becomes second nature."
char = sorted(set(text))
print("text:", text)
print("Unique characters:", char)

char_to_idx = {}
for i, ch in enumerate(char):
    char_to_idx[ch] = i

idx_to_char = {}
for ch, i in char_to_idx.items():
    idx_to_char[i] = ch

print("     char_to_idx:", char_to_idx)
print("     idx_to_char:", idx_to_char)

tokenizer = {
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char
}

print("tokenizer:", tokenizer)


bigram_counts = np.zeros((len(char), len(char)))

tokens = [char_to_idx[ch] for ch in text]

for i in range(len(tokens)-1):
    bigram_counts[tokens[i], tokens[i+1]] += 1
    
print("bigram_counts:", bigram_counts)

def softmax(x):
    exp_x= np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

bigram_probs = np.zeros_like(bigram_counts)
for i in range(len(char)):
    rows = bigram_counts[i]
    if rows.sum() > 0:
        bigram_probs[i] = softmax(rows)

print("bigram_probs:", bigram_probs)

# Visualize softmax probabilities
plt.figure(figsize=(12, 10))
sns.heatmap(bigram_probs[:15, :15], 
            xticklabels=char[:15], 
            yticklabels=char[:15],
            annot=True,           # Show numbers in cells
            fmt='.3f',           # Format as 3 decimal places
            cmap='Greens',        # Use green color scheme for probabilities
            cbar_kws={'label': 'Probability'})  # Label colorbar

plt.title('Bigram Probability Matrix (After Softmax)\n(Probability of each character following another)', fontsize=14, pad=20)
plt.xlabel('Second Character (what follows)', fontsize=12)
plt.ylabel('First Character (what comes before)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("bigram_probs_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 Probability heatmap saved as 'bigram_probs_heatmap.png'")
print("🔍 Each row sums to 1.0 (100%) - these are probabilities!")
print("📈 Darker green = higher probability of that character pair")

# Show example of how softmax works for one character
example_char = ' '  # space character
if example_char in char_to_idx:
    idx = char_to_idx[example_char]
    raw_counts = bigram_counts[idx]
    probs = bigram_probs[idx]
    
    print(f"\n📝 Example: Character '{example_char}' (index {idx})")
    print(f"Raw counts: {raw_counts[:10]}")  # Show first 10
    print(f"Probabilities: {probs[:10]}")   # Show first 10
    print(f"Sum of probabilities: {probs.sum():.6f} (should be 1.0)")

def generation(start_char, max_length):
    start_idx = char_to_idx[start_char]
    current_idx = start_idx
    result = [start_char]

    for _ in range(max_length - 1):
        next_probs = bigram_probs[current_idx]
        next_idx = np.random.choice(len(char), p=next_probs)
        result.append(idx_to_char[next_idx])
        current_idx = next_idx
    
    return ''.join(result)

print("generation:", generation("p", 100))

# Create a better heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(bigram_counts[:15, :15], 
            xticklabels=char[:15], 
            yticklabels=char[:15],
            annot=True,           # Show numbers in cells
            fmt='g',             # Format as integers
            cmap='Blues',        # Use blue color scheme
            cbar_kws={'label': 'Frequency'})  # Label colorbar

plt.title('Bigram Frequency Matrix\n(How often each character follows another)', fontsize=14, pad=20)
plt.xlabel('Second Character (what follows)', fontsize=12)
plt.ylabel('First Character (what comes before)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("bigram_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()  # Also display the plot

print("\n📊 Heatmap saved as 'bigram_heatmap.png'")
print("🔍 The heatmap shows how often each character (row) is followed by another character (column)")
print("📈 Darker colors = higher frequency")