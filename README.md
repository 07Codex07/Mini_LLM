Here is the updated README:

---

# MiniLLM 🧠

> Building a Large Language Model from scratch. No libraries. No shortcuts. Just code and understanding.

---

## What is this?

Most people use HuggingFace, LangChain, or OpenAI APIs and never understand what is actually happening underneath.

This is my attempt to fix that, for myself.

I am building a complete LLM from scratch, every single component, by hand, until it can hold a conversation on its own. No magic. No abstractions. Just NumPy, math, and first principles.

---

## 🎯 End Goal

A working language model that can reply to prompts naturally, built entirely without ML libraries. No transformers library. No PyTorch autograd. Just raw understanding.

---

## 🗺️ Roadmap

```
Tokenization ✅  →  Embeddings 🔄  →  Attention ⏳  →  Transformer ⏳  →  Training ⏳  →  Inference ⏳
```

### ✅ Stage 1 — Tokenization (Done)
- Character level tokenization
- Bigram model and probability matrix
- Why character level fails at scale
- 📝 [Read the article](your link here) — Published in Towards AI

### 🔄 Stage 2 — Embeddings (Current)
- Converting tokens to vectors
- Building an embedding layer from scratch
- Understanding what embeddings actually represent

### ⏳ Stage 3 — Attention
- Self attention from scratch
- Multi head attention
- Positional encoding

### ⏳ Stage 4 — Transformer Block
- Feed forward layers
- Layer normalization
- Residual connections

### ⏳ Stage 5 — Training Loop
- Forward pass
- Backpropagation from scratch
- Gradient descent by hand

### ⏳ Stage 6 — Inference
- Text generation
- Temperature and sampling
- Making it actually reply like an LLM

---

## 📊 Progress

| Stage | Status |
|---|---|
| Character Tokenization + Bigram Model | ✅ Done |
| Embeddings | 🔄 In Progress |
| Attention | ⏳ Not Started |
| Transformer Block | ⏳ Not Started |
| Training Loop | ⏳ Not Started |
| Inference | ⏳ Not Started |

---

## 💡 Why from scratch?

Because using a library tells you **what** to do.

Building from scratch tells you **why** it works.

Every stage of this project forced me to understand something I thought I already knew. That is the whole point.

---

## 📚 Article Series

Documenting every stage as I build it.

| Stage | Article | Publication |
|---|---|---|
| Tokenization | [Building an LLM From Scratch. Here's Where It All Starts.](your link here) | Towards AI |
| Embeddings | Coming soon | — |
| Attention | Coming soon | — |

---

## 🛠️ Stack

```python
dependencies = ["numpy", "curiosity", "patience"]
```

---

## ⚠️ Note

This is a learning project. Expect mistakes, rewrites, and honest documentation of what went wrong. That is the point.

---

*If you are also building from scratch, feel free to follow along or reach out.*

---

Replace `your link here` with your Towards AI article link and push.
