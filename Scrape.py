import http.client
import re
import time
import urllib.error
import urllib.request

# Gutenberg is friendlier with a real UA and brief pauses between large downloads.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; miniLLM-corpus/1.0; +https://www.gutenberg.org/policy/robot_access)"
)
_MAX_RETRIES = 6
_BASE_DELAY_S = 2.0
_PAUSE_BETWEEN_BOOKS_S = 2.0

# Project Gutenberg UTF-8 plain text — Dostoyevsky works referenced on PG
EBOOK_IDS = [ # Crime and Punishment
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


def _open_url(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_err: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (
            http.client.RemoteDisconnected,
            urllib.error.URLError,
            ConnectionResetError,
            TimeoutError,
            OSError,
        ) as e:
            last_err = e
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_DELAY_S * (2**attempt)
            print(f"  (retry {attempt + 1}/{_MAX_RETRIES - 1} after {e!r}; sleeping {wait:.0f}s)")
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def fetch_book(ebook_id: int) -> str:
    url = gutenberg_txt_url(ebook_id)
    raw = _open_url(url, timeout=180).decode("utf-8-sig")
    return strip_project_gutenberg_boilerplate(raw)


def main() -> None:
    parts: list[str] = []
    for i, ebook_id in enumerate(EBOOK_IDS):
        print(f"Fetching {ebook_id}...")
        text = fetch_book(ebook_id)
        parts.append(text)
        print(f"  -> {len(text)} characters")
        if i + 1 < len(EBOOK_IDS):
            time.sleep(_PAUSE_BETWEEN_BOOKS_S)

    corpus = "\n\n\n".join(parts)
    with open("corpus.txt", "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"corpus.txt length: {len(corpus)} characters")
    print(f"Sample: {corpus[:500]}")


if __name__ == "__main__":
    main()
