# MiniLLM 🧠

<img width="1440" height="1040" alt="image" src="https://github.com/user-attachments/assets/05b3d8e9-bffe-418b-bb1f-c2d7e6eb4080" />

> No libraries. No shortcuts. Every component built by hand until it can hold a conversation on its own.

---

## What is this

Most people call `model.fit()` and move on. I wanted to know what happens inside that call.

So I started from zero. No PyTorch. No HuggingFace. No LangChain. Just Python, NumPy, and first principles. Every component built by hand until I understood exactly why it works, not just that it works.

This repo documents that entire journey, including the parts that broke.

---

## 🎯 End goal

A language model that can hold a conversation on its own. Built entirely from scratch. When it works, every single number it produces will have been computed by code I wrote and understood line by line.

---

## 🗺️ The build

```
Tokenization ✅  →  Embeddings ✅  →  Attention ⏳  →  Transformer ⏳  →  Training ⏳  →  Inference ⏳
```

---

### ✅ Stage 1 — Tokenization

Character level tokenization from scratch. Built a bigram model, probability matrix, and text generation using nothing but NumPy.

**What broke:** Applied softmax to raw frequency counts instead of plain normalization. Softmax was designed for logits, not counts. Small bug, important lesson.

```python
# wrong
bigram_probs[i] = softmax(bigram_counts[i])

# correct
bigram_probs[i] = bigram_counts[i] / bigram_counts[i].sum()
```

**Key finding:** Character level tokenization fails at scale. A 40 token sentence means 1600 attention operations. That is why BPE exists.

📝 [Read the full breakdown](https://pub.towardsai.net/building-an-llm-from-scratch-heres-where-it-all-starts-071d7f1ab870) — Published in Towards AI

---

### ✅ Stage 2 — Word Embeddings

Word2Vec Skip-Gram with Negative Sampling, built from scratch. Trained on nearly 1 million words across 5 Dostoevsky novels. 20 epochs. 30+ hours on a laptop. No GPU.

**Corpus:**
- The Brothers Karamazov
- The Idiot
- The Possessed
- Notes from the Underground
- Short Stories

**What the model learned without being told:**

```
crime vs punishment:  0.8778  ✅
crime vs murder:      0.8408  ✅
crime vs guilt:       0.7630  ✅
crime vs love:        0.5965  ✅ lower, correct
crime vs happy:       0.6787  ✅ lower, correct
```

**Most remarkable finding:** Crime and Punishment was never in the training data. The model learned that crime and punishment belong together purely from context in other novels. Nobody programmed that relationship. It emerged.

**What broke and why:**

Single embedding table caused gradient collision. When the same word plays both center and context roles, gradients from both interfere during backpropagation. Every word started looking similar. Fixed with two separate matrices.

Uniform negative sampling let common words like "the" dominate. Fixed with frequency based sampling raised to 0.75 power, exactly as the original Word2Vec paper describes.

Learning rate too high caused embedding collapse. Reduced from 0.01 to 0.005.

📝 Article coming soon — Towards AI

---

### ⏳ Stage 3 — Self Attention

In progress.

---

### ⏳ Stage 4 — Transformer Block

Not started.

---

### ⏳ Stage 5 — Training Loop

Not started.

---

### ⏳ Stage 6 — Inference

Not started.

---

## 📊 Progress

| Stage | Status | Article |
|---|---|---|
| Tokenization | ✅ Done | [Read](https://pub.towardsai.net/building-an-llm-from-scratch-heres-where-it-all-starts-071d7f1ab870) |
| Embeddings | ✅ Done | Coming soon |
| Attention | ⏳ Not started | — |
| Transformer Block | ⏳ Not started | — |
| Training Loop | ⏳ Not started | — |
| Inference | ⏳ Not started | — |

---

## 📁 Structure

```
Mini_LLM/
├── Tokenizer.py                     # character level bigram model
├── embeddings.py                    # word2vec skip-gram from scratch
├── evaluate.py                      # cosine similarity evaluation
├── corpus.txt                       # ~1M words, 5 Dostoevsky novels
├── outputs/
│   ├── bigram_heatmap.png
│   ├── bigram_probs_heatmap.png
│   ├── embeddings_before_training.png
│   └── embeddings_after_training.png
├── requirements.txt
└── README.md
```

---

## 🛠️ Stack

```python
dependencies = ["numpy", "matplotlib", "seaborn", "sklearn", "curiosity", "patience"]
```

No PyTorch. No TensorFlow. No HuggingFace. That is the point.

---

## 📚 Lessons so far

Things I thought I understood before building this that I actually did not:

- Softmax is for logits, not frequency counts
- A single embedding table causes gradient interference
- Embedding collapse is real and subtle
- 1 million words is nothing. Google trained on 100 billion.
- Pure numpy loops at scale will humble you

---

## ⚠️ Honest note

This is a learning project. The embeddings are not perfect. The tokenizer is basic. Some decisions were wrong and got fixed. All of that is documented intentionally because that is how actual understanding develops.

---

## 📚 Follow along

Every stage gets a detailed article with code and honest documentation of what went wrong. Links in the progress table above.

*Built by [Vinayak Sahu](https://www.linkedin.com/in/vinayak-sahu-8999a9259)*
