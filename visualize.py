import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

embedding_table = np.load("trained_embeddings.npy")
word_to_idx = np.load("word_to_idx.npy", allow_pickle=True).item()

words_to_plot = ["crime", "murder", "guilt", "punishment", "police",
                 "love", "happy", "innocent", "death", "money",
                 "idiot", "atheist", "school", "university", "monastery"]

vectors = []
labels = []

for word in words_to_plot:
    if word in word_to_idx:
        vectors.append(embedding_table[word_to_idx[word]])
        labels.append(word)

vectors = np.array(vectors)
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

plt.figure(figsize=(12, 9))
plt.scatter(reduced[:, 0], reduced[:, 1], color='steelblue', s=100)

for i, label in enumerate(labels):
    plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=12, ha='right')

plt.title("Word Embeddings AFTER Training\n20 Epochs on 5 Dostoevsky Novels", fontsize=14)
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embeddings_after_training.png", dpi=300)
plt.show()
print("Saved.")