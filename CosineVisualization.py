import numpy as np
from embeddings import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce 32 dimensions to 2 for visualization
words_to_plot = ["crime", "punishment", "murder", "guilt", "police", "love", "happy", "innocent", "death", "money"]

vectors = []
labels = []

for word in words_to_plot:
    if word in word_to_idx:
        vectors.append(input_embeddings[word_to_idx[word]])
        labels.append(word)

vectors = np.array(vectors)

# PCA to reduce to 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], color='steelblue', s=100)

for i, label in enumerate(labels):
    plt.annotate(label, (reduced[i, 0], reduced[i, 1]), 
                fontsize=12, ha='right')

plt.title("Word Embeddings Before Training (Random)\nWords are scattered randomly", fontsize=14)
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embeddings_before_training.png", dpi=300)
plt.show()
print("Plot saved.")