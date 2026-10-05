# MiniLLM

Most people call `model.fit()` and move on. I wanted to know what's actually happening inside that call. So I'm building a language model from zero — no PyTorch, no HuggingFace, no autodiff, just NumPy and whatever math I had to relearn along the way.

I write each stage up properly on [Medium](https://medium.com/@vinayak1672006) as I finish it. This README is the short version, including what's happened since the last article.

**Where it's at:** tokenization, embeddings, attention, and a working 3-layer transformer are done and trained. Generation works, in the sense that real text comes out — just not good text past a sentence or two. The original plan was "something that can hold a conversation." I don't think that's realistic in pure NumPy on a laptop, so I've stopped chasing it. Current goal: understand every piece, keep the numbers honest.

## The model

3-layer decoder-only transformer, pre-LayerNorm, 192-dim embeddings, 4 heads, GELU feed-forward, 64-token context, weight-tied output layer. ~2.5M parameters.

## Data

67 Gutenberg books, 11.27M words, BPE I wrote myself (6,000 merges, vocab 6,089, kept case and punctuation this time — the embeddings stage had stripped both). 13.46M tokens total, last 5% held out and never trained on.

## Training

15 epochs, ~15–20 hours on a 12-thread laptop CPU. All NumPy, no GPU — see the compute note below for why that's worth saying explicitly.

Across the whole project it adds up to roughly two days of CPU time — the Word2Vec embeddings alone took 30+ hours, and that's before counting the earlier transformer runs that didn't work.

## Results

| Model | Test perplexity |
|---|---|
| Uniform | 6089.0 |
| Unigram | 1140.5 |
| Bigram (interpolated) | 260.8 |
| Trigram (interpolated) | 221.4 |
| Transformer | 133.2 |

Each step shows up clearly: counting single words gets you to 1140, counting pairs gets you to 261, counting triples gets you to 221, and actual attention over 64 tokens of context gets you to 133.

> Sherlock Holmes." What an excellent fellow that is doing?" he asked, calling to him with his usual attentive vehicle and resurrections, as if looking to him," You see, dear sir, I tell

Grammar and basic sentence structure are mostly right; meaning drifts after a clause or two, which is about right for 2.5M params on 13M tokens. Visible bug in that sample: quotes stick to the previous word — detokenizer issue, not the model, still unfixed.

## The part not on Medium yet

After attention worked on its own, the full stack got stuck. Top-5 predictions for completely different contexts were all the same — *the, and, to, of, a*. Not undertrained, just stuck: that pattern never improved across three separate runs. My understanding of why — I didn't run a controlled experiment to isolate this, so take it as my best read of it, not a confirmed diagnosis — is that plain SGD with no warmup is a specifically bad combination for transformers: the tied output layer learns the easy "predict word frequency" solution fast, while attention and the FFN get weaker signal and a high LR with no warmup can strand them before they learn anything real. Switching to Adam plus warmup fixed it outright.

Once training worked, I wanted a bigger, more varied corpus than five Dostoevsky novels, which meant more compute than a laptop CPU was going to give me comfortably. I benchmarked CuPy on a Colab GPU as a side experiment before deciding what to actually do: 18 examples/sec on CPU at the time (before the float32 and matmul fixes below) vs. 246/sec on GPU, about 13.7x — bigger than the 5.7x speedup on a smaller config, because larger matmuls amortize kernel-launch overhead better. That benchmark is real, but it's not what trained the model in the results above — the run that actually produced the 133.2 result stayed on CPU. Switching to float32 properly and fixing the matmul shape (below) was enough to get a step down to 4–6 seconds, which made a 15–20 hour CPU run realistic without needing to move off NumPy. The benchmark script isn't in this repo; the training code here is CPU-only.

## What broke, the real list

- **Softmax on raw counts, not frequencies.** Softmax exponentiates, so a count of 100 vs. 50 turns into a probability ratio of roughly e⁵⁰ — the most frequent pair swallows almost all the probability instead of just being somewhat more likely. Plain normalization doesn't have that problem.
- **One embedding table, two roles.** Center and context gradients collided during training, and the symptom was that every word's embedding started looking similar to every other word's — not useful for telling anything apart. Fixed with two separate tables, one per role.
- **Uniform negative sampling** let common words dominate. Fixed with frequency^0.75, straight from the Word2Vec paper.
- **The SGD collapse above.**
- **`glob("transformer_layer*.npz")` also matched `*_best.npz`** — a 2-layer model silently ran as 4. No crash, just subtly wrong output.
- **An updated tokenizer loaded against an old checkpoint.** Case/punctuation got silently dropped — "The" became "he."
- **An earlier 57.5 perplexity number, on the Dostoevsky corpus, was measured on text the model had already trained on** — it had trained on windows from the whole corpus, including the slice I scored it on, so it isn't a real result. Separately, random sliding windows one token apart share 63 of 64 tokens, which made the training script's own validation loss during training basically meaningless as a signal, even where it wasn't literally scoring on seen text. Both got fixed with a contiguous held-out block at the end of the corpus that training never touches.
- **float64 kept creeping back in** — NumPy defaults to it, and even `np.sqrt(192)` can silently upcast float32 arrays back.
- **Slow 3D matmuls, plus a slow GELU.** `(256, 64, 192) @ W` ran as 256 separate small matmuls instead of one big one — flattening to 2D first helped a lot. `x**3` in GELU also used NumPy's general power function, which is slower than `x*x*x`. Mostly from those two fixes together, a training step went from 8.56s to 4.24s.

## Known issues

- Quote-spacing bug in the detokenizer, unfixed
- No confirmed reason for the train/validation gap — best guess is an out-of-distribution held-out tail, unverified
- The finite-difference gradient check isn't in the repo as a runnable test yet, so I'm not quoting a number for it until it is

## Stack

NumPy, matplotlib, seaborn, scikit-learn (visualization only, in `CosineVisualization.py` and `visualize.py`). No PyTorch, no TensorFlow, no autodiff — that's the actual point.

---

Built by Vinayak Sahu. MIT license.
