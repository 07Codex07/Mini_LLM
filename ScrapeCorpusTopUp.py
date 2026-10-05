# adds more books to corpus_v2.txt until it reaches TARGET_WORDS.
# skips ids already in corpus_v2_manifest.json (or ALREADY_FETCHED on the first run)

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request

_USER_AGENT = (
    "Mozilla/5.0 (compatible; miniLLM-corpus/2.1; +https://www.gutenberg.org/policy/robot_access)"
)
_MAX_RETRIES = 6
_BASE_DELAY_S = 2.0
_PAUSE_BETWEEN_BOOKS_S = 2.0

TARGET_WORDS = 10_000_000
OUTPUT_FILE = "corpus_v2.txt"
MANIFEST_FILE = "corpus_v2_manifest.json"

# first scrape got 34 books, 3,394,624 words.
# 1661 (Sherlock Holmes) failed, probably just the network, so it's not in this list and gets retried
ALREADY_FETCHED = [11, 1342, 76, 74, 2852, 120, 43, 84, 345, 98, 1400, 730, 55, 36, 35,
                   2701, 158, 105, 1260, 768, 174, 45, 215, 910, 829, 521, 16, 113, 2591, 203]

# original list + bigger books to get to 10M. already fetched ones get skipped
BOOKS = [
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
    (1342, "Pride and Prejudice", "Jane Austen"),
    (76, "Adventures of Huckleberry Finn", "Mark Twain"),
    (74, "The Adventures of Tom Sawyer", "Mark Twain"),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (2852, "The Hound of the Baskervilles", "Arthur Conan Doyle"),
    (120, "Treasure Island", "Robert Louis Stevenson"),
    (43, "The Strange Case of Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson"),
    (84, "Frankenstein", "Mary Shelley"),
    (345, "Dracula", "Bram Stoker"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (1400, "Great Expectations", "Charles Dickens"),
    (46, "A Christmas Carol", "Charles Dickens"),
    (730, "Oliver Twist", "Charles Dickens"),
    (55, "The Wonderful Wizard of Oz", "L. Frank Baum"),
    (36, "The War of the Worlds", "H. G. Wells"),
    (35, "The Time Machine", "H. G. Wells"),
    (5230, "The Invisible Man", "H. G. Wells"),
    (2701, "Moby Dick", "Herman Melville"),
    (158, "Emma", "Jane Austen"),
    (161, "Sense and Sensibility", "Jane Austen"),
    (105, "Persuasion", "Jane Austen"),
    (1260, "Jane Eyre", "Charlotte Bronte"),
    (768, "Wuthering Heights", "Emily Bronte"),
    (174, "The Picture of Dorian Gray", "Oscar Wilde"),
    (514, "Little Women", "Louisa May Alcott"),
    (45, "Anne of Green Gables", "L. M. Montgomery"),
    (215, "The Call of the Wild", "Jack London"),
    (910, "White Fang", "Jack London"),
    (829, "Gulliver's Travels", "Jonathan Swift"),
    (521, "Robinson Crusoe", "Daniel Defoe"),
    (16, "Peter Pan", "J. M. Barrie"),
    (113, "The Secret Garden", "Frances Hodgson Burnett"),
    (2591, "Grimms' Fairy Tales", "Brothers Grimm"),
    (203, "Uncle Tom's Cabin", "Harriet Beecher Stowe"),
    # bigger books
    (2000, "Don Quixote", "Miguel de Cervantes"),
    (1080, "A Modest Proposal", "Jonathan Swift"),
    (25344, "The Scarlet Letter", "Nathaniel Hawthorne"),
    (33, "The Scarlet Pimpernel", "Baroness Emma Orczy"),
    (244, "A Study in Scarlet", "Arthur Conan Doyle"),
    (2097, "The Sign of the Four", "Arthur Conan Doyle"),
    (863, "The Return of Sherlock Holmes", "Arthur Conan Doyle"),
    (1023, "Bleak House", "Charles Dickens"),
    (580, "The Pickwick Papers", "Charles Dickens"),
    (766, "David Copperfield", "Charles Dickens"),
    (883, "Our Mutual Friend", "Charles Dickens"),
    (700, "The Old Curiosity Shop", "Charles Dickens"),
    (967, "Nicholas Nickleby", "Charles Dickens"),
    (599, "Vanity Fair", "William Makepeace Thackeray"),
    (1257, "The Three Musketeers", "Alexandre Dumas"),
    (1184, "The Count of Monte Cristo", "Alexandre Dumas"),
    (135, "Les Miserables", "Victor Hugo"),
    (2413, "Notre-Dame de Paris", "Victor Hugo"),
    (86, "A Connecticut Yankee in King Arthur's Court", "Mark Twain"),
    (119, "A Princess of Mars", "Edgar Rice Burroughs"),
    (78, "Tarzan of the Apes", "Edgar Rice Burroughs"),
    (164, "Twenty Thousand Leagues Under the Sea", "Jules Verne"),
    (103, "Around the World in Eighty Days", "Jules Verne"),
    (18857, "A Journey to the Centre of the Earth", "Jules Verne"),
    (2488, "Anna Karenina", "Leo Tolstoy"),
    (1399, "War and Peace", "Leo Tolstoy"),
    (205, "Walden", "Henry David Thoreau"),
    (219, "Heart of Darkness", "Joseph Conrad"),
    (526, "Siddhartha", "Hermann Hesse"),
    (58585, "The Great Gatsby", "F. Scott Fitzgerald"),
    (6130, "The Iliad", "Homer"),
    (1727, "The Odyssey", "Homer"),
]

_START_LINE = re.compile(
    r"^\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK [^\n]+ \*{3}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_END_LINE = re.compile(
    r"^\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK [^\n]+ \*{3}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# "Title: X" line first, else the newer "The Project Gutenberg eBook of X, by Y" header
_TITLE_LINE = re.compile(r"Title:\s*(.+?)\s*[\r\n]", re.IGNORECASE)
_HEADER_TITLE_LINE = re.compile(
    r"Project Gutenberg\s*eBook\s+of\s+(.+?)(?:,?\s+by\s+.+)?\s*[\r\n]",
    re.IGNORECASE,
)


def gutenberg_txt_url(ebook_id: int) -> str:
    return f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt"


def gutenberg_txt_url_fallback(ebook_id: int) -> str:
    return f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt"


def gutenberg_txt_url_fallback2(ebook_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"


def strip_project_gutenberg_boilerplate(raw: str) -> tuple[str, str]:
    header = raw[:3000]  # title is always near the top
    title_m = _TITLE_LINE.search(header)
    if title_m:
        detected_title = title_m.group(1).strip()
    else:
        header_m = _HEADER_TITLE_LINE.search(header)
        detected_title = header_m.group(1).strip() if header_m else "(title not found)"

    start_m = _START_LINE.search(raw)
    end_m = _END_LINE.search(raw)
    if not start_m or not end_m:
        raise ValueError("Could not find Project Gutenberg START/END marker lines")
    return raw[start_m.end() : end_m.start()].strip(), detected_title


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
            urllib.error.HTTPError,
            ConnectionResetError,
            TimeoutError,
            OSError,
        ) as e:
            last_err = e
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_DELAY_S * (2**attempt)
            print(f"    (retry {attempt + 1}/{_MAX_RETRIES - 1} after {e!r}; sleeping {wait:.0f}s)")
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def fetch_book(ebook_id: int) -> tuple[str, str]:
    last_err: Exception | None = None
    for url_fn in (gutenberg_txt_url, gutenberg_txt_url_fallback, gutenberg_txt_url_fallback2):
        try:
            raw = _open_url(url_fn(ebook_id), timeout=180).decode("utf-8-sig")
            return strip_project_gutenberg_boilerplate(raw)
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def load_manifest() -> set[int]:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f)["fetched_ids"])
    return set(ALREADY_FETCHED)


def save_manifest(fetched_ids: set[int]) -> None:
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump({"fetched_ids": sorted(fetched_ids)}, f, indent=2)


def current_word_count() -> int:
    # count from the file itself instead of a hardcoded number
    if not os.path.exists(OUTPUT_FILE):
        return 0
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        return len(f.read().split())


def main() -> None:
    fetched_ids = load_manifest()
    total_words = current_word_count()

    to_fetch = [(eid, title, author) for eid, title, author in BOOKS if eid not in fetched_ids]
    print(f"Already have {len(fetched_ids)} books, {total_words:,} words.")
    print(f"{len(to_fetch)} new candidates queued; need {max(0, TARGET_WORDS - total_words):,} more words.\n")

    succeeded, failed, mismatched = [], [], []
    new_parts: list[str] = []

    for i, (ebook_id, expected_title, author) in enumerate(to_fetch):
        if total_words >= TARGET_WORDS:
            print(f"Reached {TARGET_WORDS:,} words — stopping early.")
            break

        print(f"Fetching #{ebook_id} (expecting: '{expected_title}' by {author})...")
        try:
            text, detected_title = fetch_book(ebook_id)
        except Exception as e:
            print(f"  FAILED: {e!r} — skipping")
            failed.append((ebook_id, expected_title))
            continue

        word_count = len(text.split())
        title_matches = expected_title.lower()[:15] in detected_title.lower()
        flag = "" if title_matches else "  <-- MISMATCH, check this ID"
        print(f"  -> fetched '{detected_title}' | {word_count:,} words{flag}")

        if not title_matches:
            mismatched.append((ebook_id, expected_title, detected_title))

        new_parts.append(text)
        total_words += word_count
        succeeded.append(ebook_id)
        fetched_ids.add(ebook_id)
        print(f"  Running total: {total_words:,} words")

        if i + 1 < len(to_fetch):
            time.sleep(_PAUSE_BETWEEN_BOOKS_S)

    if new_parts:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for part in new_parts:
                f.write("\n\n\n" + part)
    save_manifest(fetched_ids)

    print()
    print("=" * 60)
    print(f"Appended {len(succeeded)} new books. Total now: {total_words:,} words")
    if failed:
        print(f"Failed (skipped): {failed}")
    if mismatched:
        print("TITLE MISMATCHES — verify these IDs manually:")
        for eid, expected, detected in mismatched:
            print(f"  #{eid}: expected '{expected}', got '{detected}'")
    if total_words < TARGET_WORDS:
        print(f"\nStill short of {TARGET_WORDS:,} by {TARGET_WORDS - total_words:,} words.")
        print("Add more (ebook_id, title, author) entries to BOOKS and re-run.")
    else:
        print(f"\nTarget reached: {total_words:,} words in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
