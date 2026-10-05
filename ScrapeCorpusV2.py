import http.client
import re
import time
import urllib.error
import urllib.request

# real user agent + pauses between books, gutenberg blocks you otherwise
_USER_AGENT = (
    "Mozilla/5.0 (compatible; miniLLM-corpus/2.0; +https://www.gutenberg.org/policy/robot_access)"
)
_MAX_RETRIES = 6
_BASE_DELAY_S = 2.0
_PAUSE_BETWEEN_BOOKS_S = 2.0

TARGET_WORDS = 10_000_000
OUTPUT_FILE = "corpus_v2.txt"

# (ebook_id, expected title, author). title is checked after download because a wrong id
# just gives you a different book with no error
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
_TITLE_LINE = re.compile(r"Title:\s*(.+?)\s*[\r\n]", re.IGNORECASE)
# newer headers have this instead of a "Title:" line
_EBOOK_OF_LINE = re.compile(r"Project Gutenberg\s*eBook\s*of\s*(.+?),?\s*by\s", re.IGNORECASE)


def gutenberg_txt_url(ebook_id: int) -> str:
    return f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt"


def gutenberg_txt_url_fallback(ebook_id: int) -> str:
    # some older books don't have the -0 file
    return f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt"


def strip_project_gutenberg_boilerplate(raw: str) -> tuple[str, str]:
    # returns (text, title found in the header)
    header = raw[:3000]  # title is always near the top
    title_m = _TITLE_LINE.search(header)
    if title_m:
        detected_title = title_m.group(1).strip()
    else:
        ebook_m = _EBOOK_OF_LINE.search(header)
        detected_title = ebook_m.group(1).strip() if ebook_m else "(no Title: line found)"

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
    try:
        raw = _open_url(gutenberg_txt_url(ebook_id), timeout=180).decode("utf-8-sig")
    except Exception:
        raw = _open_url(gutenberg_txt_url_fallback(ebook_id), timeout=180).decode("utf-8-sig")
    return strip_project_gutenberg_boilerplate(raw)


def main() -> None:
    parts: list[str] = []
    total_words = 0
    succeeded, failed, mismatched = [], [], []

    for i, (ebook_id, expected_title, author) in enumerate(BOOKS):
        if total_words >= TARGET_WORDS:
            print(f"Reached {TARGET_WORDS:,} words — stopping early, {len(BOOKS) - i} books unused.")
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

        parts.append(text)
        total_words += word_count
        succeeded.append(ebook_id)
        print(f"  Running total: {total_words:,} words")

        if i + 1 < len(BOOKS):
            time.sleep(_PAUSE_BETWEEN_BOOKS_S)

    corpus = "\n\n\n".join(parts)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(corpus)

    print()
    print("=" * 60)
    print(f"Wrote {OUTPUT_FILE}: {len(corpus):,} characters, {total_words:,} words")
    print(f"Books fetched: {len(succeeded)}/{len(BOOKS)}")
    if failed:
        print(f"Failed (skipped): {failed}")
    if mismatched:
        print(f"TITLE MISMATCHES — verify these IDs manually:")
        for eid, expected, detected in mismatched:
            print(f"  #{eid}: expected '{expected}', got '{detected}'")
    if total_words < TARGET_WORDS:
        print(f"\nShort of the {TARGET_WORDS:,}-word target by {TARGET_WORDS - total_words:,} words.")
        print("Add more (ebook_id, title, author) entries to BOOKS and re-run.")


if __name__ == "__main__":
    main()
