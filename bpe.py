import re
import unicodedata
from collections import Counter, defaultdict

END = "</w>"  # end of word marker, so "cat" and the "cat" in "category" stay different

_BRACKET_TAG = re.compile(r"\[[^\]\n]*\]")  # [Illustration], [Footnote 1] etc

# contractions, words, numbers, punctuation. keeps case and punctuation now -
# lowercasing everything meant the model could never learn where sentences end
_PRETOKEN = re.compile(r"[A-Za-z]+'[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\w\s]+")

_PUNCT_NORMALIZE = {
    "’": "'", "‘": "'",   # curly quotes -> straight
    "“": '"', "”": '"',
    "–": "-", "—": "-",   # all dashes -> "-"
    "…": "...",
}

_NO_SPACE_BEFORE = set(".,!?;:')]}\"")  # join_words: no space before these


def clean_words(text):
    # used for the vocab, the training text and the test input - all three have to match
    for src, dst in _PUNCT_NORMALIZE.items():
        text = text.replace(src, dst)
    # strip accents (café -> cafe), otherwise the accent becomes its own token mid-word
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _BRACKET_TAG.sub(" ", text)
    return _PRETOKEN.findall(text)


def join_words(words):
    # " ".join would give "Hello , world ." since punctuation is its own token
    out = []
    for w in words:
        if out and w and w[0] in _NO_SPACE_BEFORE:
            out[-1] = out[-1] + w
        else:
            out.append(w)
    return " ".join(out)


def get_word_freqs(text):
    return Counter(clean_words(text))


def _word_to_symbols(word):
    return list(word) + [END]


def _merge_word(symbols, pair):
    merged = "".join(pair)
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def _init_pair_stats(word_splits, word_freqs):
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for w, symbols in word_splits.items():
        freq = word_freqs[w]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(w)
    return pair_counts, pair_to_words


def train_bpe(word_freqs, num_merges, verbose=True):
    word_splits = {w: _word_to_symbols(w) for w in word_freqs}
    pair_counts, pair_to_words = _init_pair_stats(word_splits, word_freqs)

    merges = []
    for step in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[best_pair] < 2:
            break
        merges.append(best_pair)

        # only update words that contain this pair
        for w in list(pair_to_words[best_pair]):
            old_symbols = word_splits[w]
            freq = word_freqs[w]

            for i in range(len(old_symbols) - 1):
                p = (old_symbols[i], old_symbols[i + 1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                pair_to_words[p].discard(w)

            new_symbols = _merge_word(old_symbols, best_pair)
            word_splits[w] = new_symbols

            for i in range(len(new_symbols) - 1):
                p = (new_symbols[i], new_symbols[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(w)

        pair_counts.pop(best_pair, None)

        if verbose and step % 500 == 0:
            print(f"  merge {step}/{num_merges}: {best_pair} -> '{''.join(best_pair)}'")

    vocab = set()
    for symbols in word_splits.values():
        vocab.update(symbols)
    return merges, sorted(vocab)


def build_merges_rank(merges):
    return {pair: rank for rank, pair in enumerate(merges)}


def apply_bpe_to_word(word, merges_rank):
    symbols = _word_to_symbols(word)
    while len(symbols) > 1:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        ranked = [(merges_rank[p], p) for p in pairs if p in merges_rank]
        if not ranked:
            break
        _, best_pair = min(ranked)
        symbols = _merge_word(symbols, best_pair)
    return symbols


def build_vocab_index(vocab_symbols):
    vocab_symbols = sorted(vocab_symbols)
    token_to_id = {tok: i for i, tok in enumerate(vocab_symbols)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return token_to_id, id_to_token


def encode_words(word_list, merges_rank, token_to_id):
    ids = []
    for w in word_list:
        for tok in apply_bpe_to_word(w, merges_rank):
            if tok in token_to_id:
                ids.append(token_to_id[tok])
    return ids


def decode_ids(ids, id_to_token):
    words = []
    current = ""
    for i in ids:
        tok = id_to_token.get(i, "")
        if tok.endswith(END):
            current += tok[:-len(END)]
            words.append(current)
            current = ""
        else:
            current += tok
    if current:
        words.append(current)
    return words
