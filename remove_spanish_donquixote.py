# ebook #2000 turned out to be the Spanish original of Don Quixote, not a translation.
# removes chunks 42052-47651 (split on "\n\n\n") - word count matched the scrape log (386,614).
# writes corpus_v2_clean.txt, corpus_v2.txt stays as is

IN_PATH = "corpus_v2.txt"
OUT_PATH = "corpus_v2_clean.txt"
START_CHUNK = 42052
END_CHUNK = 47651  # exclusive


def main() -> None:
    with open(IN_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = raw.split("\n\n\n")
    removed = chunks[START_CHUNK:END_CHUNK]
    removed_words = sum(len(c.split()) for c in removed)

    # make sure it's still the spanish part before removing anything
    removed_preview = " ".join(removed[0].split()[:6]) if removed else ""
    if "ingenioso" not in removed_preview.lower() and "quijote" not in " ".join(removed[:3]).lower():
        print("SANITY CHECK FAILED: the targeted range doesn't look like the Spanish")
        print("Don Quixote content anymore. Aborting without writing anything.")
        print(f"First removed chunk preview: {removed_preview!r}")
        return

    kept = chunks[:START_CHUNK] + chunks[END_CHUNK:]
    cleaned = "\n\n\n".join(kept)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Removed {len(removed)} chunks, {removed_words:,} words (the Spanish Don Quixote text).")
    print(f"Before: {len(raw.split()):,} words")
    print(f"After:  {len(cleaned.split()):,} words")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
