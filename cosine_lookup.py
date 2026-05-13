import numpy as np

embedding_table = np.load("trained_embeddings.npy")
word_to_idx = np.load("word_to_idx.npy", allow_pickle=True).item()
idx_to_word = {i: w for w, i in word_to_idx.items()}

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

test_words = ["crime", "punishment", "murder", "guilt", "police", "love", "happy"]

print("Cosine similarity after 20 epochs:\n")
print(f"crime vs punishment: {cosine_similarity(embedding_table[word_to_idx['crime']], embedding_table[word_to_idx['punishment']]):.4f}")
print(f"crime vs murder:     {cosine_similarity(embedding_table[word_to_idx['crime']], embedding_table[word_to_idx['murder']]):.4f}")
print(f"crime vs guilt:      {cosine_similarity(embedding_table[word_to_idx['crime']], embedding_table[word_to_idx['guilt']]):.4f}")
print(f"crime vs police:     {cosine_similarity(embedding_table[word_to_idx['crime']], embedding_table[word_to_idx['police']]):.4f}")
print(f"crime vs love:       {cosine_similarity(embedding_table[word_to_idx['crime']], embedding_table[word_to_idx['love']]):.4f}")
print(f"crime vs happy:      {cosine_similarity(embedding_table[word_to_idx['crime']], embedding_table[word_to_idx['happy']]):.4f}")

print("\nTop 10 most similar words to 'is':")
crime_vec = embedding_table[word_to_idx['is']]
similarities = [(w, cosine_similarity(crime_vec, embedding_table[i])) for w, i in word_to_idx.items()]
similarities.sort(key=lambda x: x[1], reverse=True)
for word, sim in similarities[:10]:
    print(f"  {word}: {sim:.4f}")