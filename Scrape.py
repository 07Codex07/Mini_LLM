import re
import urllib.request

# Project Gutenberg UTF-8 plain text — Dostoyevsky works referenced on PG
EBOOK_IDS = [
    28054,  # The Brothers Karamazov
    2638,  # The Idiot
    600,  # Notes from the Underground
    8117,  # The Possessed (The Devils)
    40745,  # Short Stories
]

_START_LINE = re.compile(
    r"^\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK [^\n]+ \*{3}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_END_LINE = re.compile(
    r"^\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK [^\n]+ \*{3}\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def gutenberg_txt_url(ebook_id: int) -> str:
    return f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt"


def strip_project_gutenberg_boilerplate(raw: str) -> str:
    start_m = _START_LINE.search(raw)
    end_m = _END_LINE.search(raw)
    if not start_m or not end_m:
        raise ValueError("Could not find Project Gutenberg START/END marker lines")
    return raw[start_m.end() : end_m.start()].strip()


def fetch_book(ebook_id: int) -> str:
    url = gutenberg_txt_url(ebook_id)
    with urllib.request.urlopen(url, timeout=120) as resp:
        raw = resp.read().decode("utf-8-sig")
    return strip_project_gutenberg_boilerplate(raw)


def main() -> None:
    parts: list[str] = []
    for ebook_id in EBOOK_IDS:
        print(f"Fetching {ebook_id}...")
        text = fetch_book(ebook_id)
        parts.append(text)
        print(f"  -> {len(text)} characters")

    corpus = "\n\n\n".join(parts)
    with open("corpus.txt", "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"corpus.txt length: {len(corpus)} characters")
    print(f"Sample: {corpus[:500]}")


if __name__ == "__main__":
    main()
