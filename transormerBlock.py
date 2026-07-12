import numpy as np

embeddingsDim = 64
seq_len = 16
d_k = 64
word_to_idx = np.load("word_to_idx.npy", allow_pickle=True).item()

idx_to_word = {i: w for w, i in word_to_idx.items()}
vocabSize = len(word_to_idx)
input_embeddings = np.random.randn(vocabSize, embeddingsDim)

def positionalEncoding(seq_len, embeddingsDim):
    matrix = np.zeros((seq_len, embeddingsDim))
    for pos in range(seq_len):
        for i in range(embeddingsDim):
            if i % 2 == 0:
                matrix[pos, i] = np.sin(pos / (10000 ** (i / embeddingsDim)))
            else:
                matrix[pos, i] = np.cos(pos / (10000 ** (i / embeddingsDim)))
    return matrix

positionalEncoding = positionalEncoding(seq_len, embeddingsDim)
print(positionalEncoding)
