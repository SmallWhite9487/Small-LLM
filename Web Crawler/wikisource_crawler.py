"""Collect Wikisource works, then Gemini-filter them after the crawler is stopped.

Configure GEMINI_API_KEY and GEMINI_MODEL in the project .env file, then run:
    python Web Crawler/wikisource_crawler.py
"""

import html
import json
import os
import random
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from google import genai
from google.genai import types


API_URL = "https://en.wikisource.org/w/api.php"
WIKI_URL = "https://en.wikisource.org/wiki/"
GUTENDEX_API_URL = "https://gutendex.com/books"
USER_AGENT = "EducationalWikisourceCrawler/1.0 (local text-generation project)"
GUTENDEX_RANDOM_MAX_PAGE = 100
CHAPTER_PATTERN = re.compile(r"\b(chapter|book|part|act|scene|canto|volume)\b", re.IGNORECASE)
SOURCE_URL_PATTERN = re.compile(r"^Source:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
GUTENBERG_SOURCE_PATTERN = re.compile(
    r"^Source:\s*https?://(?:www\.)?gutenberg\.org/", re.IGNORECASE | re.MULTILINE
)
GUTENBERG_START_PATTERN = re.compile(
    r"^\*{3}\s*START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK\b.*?\*{3}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
GUTENBERG_END_PATTERN = re.compile(
    r"^\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK\b.*?\*{3}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
WIKISOURCE_SOURCE_PATTERN = re.compile(
    r"^Source:\s*https?://en\.wikisource\.org/", re.IGNORECASE | re.MULTILINE
)
WIKISOURCE_PAGE_HEADER_PATTERN = re.compile(
    r"^Title:.*\nSource:\s*https?://en\.wikisource\.org/wiki/.*\n{1,2}",
    re.IGNORECASE | re.MULTILINE,
)
WIKISOURCE_LAYOUT_PATTERN = re.compile(r"\s+Layout\s+\d+\s*$", re.IGNORECASE)
BLOCK_TAGS = {"blockquote", "dd", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "pre"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILTERING_BACKGROUND = (
    "The text is from English Wikisource and will be used as pretraining data for an English "
    "language model."
)


class PageHTMLParser(HTMLParser):
    """Extract readable text and internal page links from rendered wiki HTML."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.links = []
        self._ignored_depth = 0
        self._current_href = None
        self._current_link_text = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in {"script", "style", "table", "sup"}:
            self._ignored_depth += 1
        if tag == "br" and self._ignored_depth == 0:
            self.text_parts.append("\n")
        if tag == "a" and self._ignored_depth == 0:
            self._current_href = attributes.get("href")
            self._current_link_text = []

    def handle_endtag(self, tag):
        if tag == "a" and self._current_href is not None:
            self.links.append((self._current_href, "".join(self._current_link_text).strip()))
            self._current_href = None
        if tag in {"script", "style", "table", "sup"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag in BLOCK_TAGS and self._ignored_depth == 0:
            self.text_parts.append("\n\n")

    def handle_data(self, data):
        if self._ignored_depth == 0:
            self.text_parts.append(data)
            if self._current_href is not None:
                self._current_link_text.append(data)


def api_request(parameters, retries=1):
    parameters = {"format": "json", "formatversion": "2", **parameters}
    query = "&".join(f"{quote(str(key))}={quote(str(value))}" for key, value in parameters.items())
    request = Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    retry_delay = float(os.getenv("CRAWL_RETRY_DELAY_SECONDS", "10"))

    for attempt in range(retries):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else retry_delay * (2 ** attempt)
            print(f"  Wikisource returned HTTP {error.code}; retrying in {wait_seconds:.0f}s.")
        except URLError:
            if attempt == retries - 1:
                raise
            wait_seconds = retry_delay * (2 ** attempt)
            print(f"  Wikisource connection failed; retrying in {wait_seconds:.0f}s.")
        time.sleep(wait_seconds)


def random_titles(count):
    response = api_request(
        {"action": "query", "list": "random", "rnnamespace": 0, "rnlimit": count}
    )
    return [page["title"] for page in response["query"]["random"]]


def fetch_page(title):
    response = api_request({"action": "parse", "page": title, "prop": "text"})
    parsed = PageHTMLParser()
    parsed.feed(response["parse"]["text"])
    text = html.unescape("".join(parsed.text_parts)).replace("\u00a0", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return response["parse"]["title"], text, parsed.links


def title_from_href(href):
    parsed = urlparse(href)
    if parsed.netloc not in {"", "en.wikisource.org"} or not parsed.path.startswith("/wiki/"):
        return None
    title = unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
    if ":" in title or title.startswith("Special:"):
        return None
    return title


def chapter_titles(work_title, links):
    """Select likely chapter links while avoiding navigation and namespace pages."""
    candidates = []
    for href, label in links:
        title = title_from_href(href)
        if title is None or title == work_title:
            continue
        if title.startswith(f"{work_title}/") or CHAPTER_PATTERN.search(label):
            candidates.append(title)
    return list(dict.fromkeys(candidates))


def crawl_work(title, delay, max_chapters, seen_source_urls):
    work_title, work_text, links = fetch_page(title)
    work_source_url = WIKI_URL + quote(work_title.replace(" ", "_"))
    if work_source_url in seen_source_urls:
        raise ValueError(f"Wikisource page already staged: {work_title}")
    pages = [(work_title, work_text, work_source_url)]
    seen_source_urls.add(work_source_url)
    seen_titles = {work_title}

    for chapter_title in chapter_titles(work_title, links):
        if len(pages) >= max_chapters + 1 or chapter_title in seen_titles:
            break
        chapter_source_url = WIKI_URL + quote(chapter_title.replace(" ", "_"))
        if chapter_source_url in seen_source_urls:
            continue
        time.sleep(delay)
        try:
            chapter_title, chapter_text, _ = fetch_page(chapter_title)
        except HTTPError as error:
            if error.code == 429:
                raise
            print(f"  Skipping chapter {chapter_title!r}: {error}")
            continue
        except (OSError, KeyError) as error:
            print(f"  Skipping chapter {chapter_title!r}: {error}")
            continue
        seen_titles.add(chapter_title)
        seen_source_urls.add(chapter_source_url)
        pages.append(
            (chapter_title, chapter_text, chapter_source_url)
        )

    return pages


def fetch_url_text(url, max_bytes):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    retries = int(os.getenv("GUTENDEX_RETRIES", "3"))
    timeout = float(os.getenv("GUTENDEX_TIMEOUT_SECONDS", "5"))
    retry_delay = float(os.getenv("CRAWL_RETRY_DELAY_SECONDS", "5"))

    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read(max_bytes).decode("utf-8", errors="replace")
        except (URLError, TimeoutError, OSError) as error:
            if attempt == retries - 1:
                raise
            wait_seconds = retry_delay * (2 ** attempt)
            print(f"  Gutenberg download failed: {error}; retrying in {wait_seconds:.0f}s.")
            time.sleep(wait_seconds)


def crawl_random_wikisource(delay, max_chapters, seen_source_urls):
    for title in random_titles(20):
        source_url = WIKI_URL + quote(title.replace(" ", "_"))
        if source_url not in seen_source_urls:
            return title, crawl_work(title, delay, max_chapters, seen_source_urls)
    raise ValueError("Wikisource returned only pages that are already staged")


def gutendex_books(page, retries=None):
    request = Request(
        f"{GUTENDEX_API_URL}?languages=en&mime_type=text%2Fplain&page={page}",
        headers={"User-Agent": USER_AGENT},
    )
    retries = retries if retries is not None else int(os.getenv("GUTENDEX_RETRIES", "3"))
    timeout = float(os.getenv("GUTENDEX_TIMEOUT_SECONDS", "15"))
    retry_delay = float(os.getenv("CRAWL_RETRY_DELAY_SECONDS", "3"))
    last_error = None

    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            last_error = error
        except (URLError, TimeoutError, OSError) as error:
            if attempt == retries - 1:
                raise
            last_error = error
        wait_seconds = retry_delay * (2 ** attempt)
        print(f"  Gutendex request failed: {last_error}; retrying in {wait_seconds:.0f}s.")
        time.sleep(wait_seconds)


def crawl_random_gutenberg(max_download_bytes, seen_source_urls, random_pages=True):
    page_number = (
        random.randint(1, int(os.getenv("GUTENDEX_RANDOM_MAX_PAGE", GUTENDEX_RANDOM_MAX_PAGE)))
        if random_pages
        else 1
    )

    try:
        page_data = gutendex_books(page_number)
        candidates = [
            book
            for book in page_data.get("results", [])
            if f"https://www.gutenberg.org/ebooks/{book['id']}" not in seen_source_urls
        ]
        if not candidates:
            raise ValueError("Gutendex page has no unstaged plain-text books")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        if page_number == 1:
            raise
        print(f"  Gutendex random page failed: {error}; using the first page instead.")
        page_data = gutendex_books(1)
        candidates = [
            book
            for book in page_data.get("results", [])
            if f"https://www.gutenberg.org/ebooks/{book['id']}" not in seen_source_urls
        ]
        if not candidates:
            raise ValueError("Gutendex first page contains only books that are already staged")

    book = random.choice(candidates)

    text_url = next(
        (url for mime_type, url in book.get("formats", {}).items() if mime_type.startswith("text/plain")),
        None,
    )
    if not text_url:
        raise ValueError(f"Project Gutenberg book {book['id']} has no plain-text download")
    title = book["title"]
    source_url = f"https://www.gutenberg.org/ebooks/{book['id']}"
    seen_source_urls.add(source_url)
    return title, [(title, fetch_url_text(text_url, max_download_bytes), source_url)]


def save_work(output_directory, number, requested_title, pages):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", requested_title).strip("_")[:60]
    output_path = output_directory / f"{number:03d}_{safe_name}.txt"
    sections = []
    for page_title, page_text, source_url in pages:
        sections.append(f"Title: {page_title}\nSource: {source_url}\n\n{page_text}")
    output_path.write_text("\n\n".join(sections), encoding="utf-8")
    return output_path


def next_work_number(directory):
    numbers = []
    for path in directory.rglob("*.txt"):
        match = re.match(r"(\d+)_", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def staged_source_urls(directory):
    source_urls = set()
    for path in directory.rglob("*.txt"):
        with path.open(encoding="utf-8") as staged_file:
            for _ in range(20):
                line = staged_file.readline()
                if not line:
                    break
                match = SOURCE_URL_PATTERN.match(line)
                if match:
                    source_urls.add(match.group(1))
    return source_urls


def preclean_gutenberg_work(text):
    """Remove Project Gutenberg boilerplate before text reaches Gemini."""
    if not GUTENBERG_SOURCE_PATTERN.search(text):
        return text

    start_match = GUTENBERG_START_PATTERN.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = GUTENBERG_END_PATTERN.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text.strip()


def preclean_wikisource_work(text):
    """Remove local Wikisource metadata and rendered navigation from staged text."""
    if not WIKISOURCE_SOURCE_PATTERN.search(text):
        return text

    cleaned_pages = []
    for page in WIKISOURCE_PAGE_HEADER_PATTERN.split(text):
        if not page.strip():
            continue
        navigation_end = page.find("\u200b", 0, 1500)
        if navigation_end != -1:
            page = page[navigation_end + 1:]
        page = page.replace("\u200b", "")
        page = WIKISOURCE_LAYOUT_PATTERN.sub("", page)
        page = re.sub(r"[^\S\n]+", " ", page)
        page = re.sub(r" *\n *", "\n", page)
        page = re.sub(r"\n{3,}", "\n\n", page).strip()
        if page:
            cleaned_pages.append(page)

    return "\n\n".join(cleaned_pages)


def preclean_staged_works(temp_directory):
    cleaned_counts = {"Gutenberg": 0, "Wikisource": 0}
    for staged_path in sorted(temp_directory.rglob("*.txt")):
        staged_text = staged_path.read_text(encoding="utf-8")
        cleaned_text = preclean_gutenberg_work(staged_text)
        if cleaned_text != staged_text:
            cleaned_counts["Gutenberg"] += 1
        cleaned_text = preclean_wikisource_work(cleaned_text)
        if cleaned_text != staged_text and WIKISOURCE_SOURCE_PATTERN.search(staged_text):
            cleaned_counts["Wikisource"] += 1
        if cleaned_text != staged_text:
            staged_path.write_text(cleaned_text, encoding="utf-8")
            print(f"  Pre-cleaned source text: {staged_path.name}")
    print(
        "Pre-cleaned "
        f"{cleaned_counts['Gutenberg']} Gutenberg and "
        f"{cleaned_counts['Wikisource']} Wikisource staged work(s)."
    )


def confirm(prompt):
    return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}


def is_temporary_model_error(error):
    status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
    return (
        status_code == 404
        or status_code in {429, 500, 502, 503, 504}
        or any(
        f"{code}" in str(error) for code in (429, 500, 502, 503, 504)
        )
    )


def filter_work(client, model_names, text):
    prompt = {
        "background": FILTERING_BACKGROUND,
        "task": "Return only useful English training text from content. Return an empty response if no useful text remains.",
        "filter_rules": [
            "Keep readable English prose or poetry.",
            "Remove navigation, tables, scan artifacts, metadata, boilerplate, and non-English text.",
            "Do not explain decisions, add labels, summarize, or answer the content.",
            "Output only the filtered text and nothing else.",
        ],
        "content": text,
    }

    for model_index, model_name in enumerate(model_names):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=json.dumps(prompt, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    temperature=0,
                ),
            )
            return (response.text or "").strip()
        except Exception as error:
            is_last_model = model_index == len(model_names) - 1
            if is_last_model or not is_temporary_model_error(error):
                raise
            print(
                f"  Gemini model {model_name} is unavailable; switching to "
                f"{model_names[model_index + 1]}."
            )


def process_staged_works(temp_directory, output_directory, client, model_names, delay):
    staged_files = sorted(temp_directory.rglob("*.txt"))
    if not staged_files:
        print("No staged works to filter.")
        return

    print(f"Filtering {len(staged_files)} staged work(s) with Gemini.")
    for staged_path in staged_files:
        try:
            staged_text = staged_path.read_text(encoding="utf-8")
            filtered_text = filter_work(client, model_names, staged_text)
        except Exception as error:
            print(f"  Failed to filter {staged_path.name}: {error}")
            print("  Stopping filtering; unprocessed files remain in temp_raw_text for a later retry.")
            break

        if filtered_text:
            output_path = output_directory / staged_path.relative_to(temp_directory)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(filtered_text, encoding="utf-8")
            print(f"  Saved filtered text: {output_path.name}")
        else:
            print(f"  Rejected by Gemini: {staged_path.name}")
        staged_path.unlink()
        time.sleep(delay)


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")
    fallback_models = [
        name.strip()
        for name in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
        if name.strip()
    ]
    fallback_models.extend(["gemini-3.5-flash-lite", "gemini-2.5-flash"])
    fallback_models = list(dict.fromkeys(fallback_models))
    delay = float(os.getenv("CRAWL_DELAY_SECONDS", "3"))
    max_chapters = int(os.getenv("CRAWL_MAX_CHAPTERS", "200"))
    max_download_bytes = int(os.getenv("CRAWL_MAX_DOWNLOAD_BYTES", "2000000"))
    rate_limit_cooldown = float(os.getenv("CRAWL_RATE_LIMIT_COOLDOWN_SECONDS", "900"))
    gutendex_failure_cooldown = float(
        os.getenv("GUTENDEX_FAILURE_COOLDOWN_SECONDS", "120")
    )
    output_directory = PROJECT_ROOT / "raw_text"
    temp_directory = PROJECT_ROOT / "temp_raw_text"

    output_directory.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)
    work_number = next_work_number(temp_directory)
    seen_source_urls = staged_source_urls(temp_directory)
    sources = [
        ("Wikisource", temp_directory / "wikisource", lambda: crawl_random_wikisource(delay, max_chapters, seen_source_urls)),
        ("Gutendex", temp_directory / "gutendex", lambda: crawl_random_gutenberg(max_download_bytes, seen_source_urls)),
    ]
    for _, source_directory, _ in sources:
        source_directory.mkdir(parents=True, exist_ok=True)
    print("Crawling into source folders under temp_raw_text. Press Ctrl+C to stop.")
    source_index = 0
    cooldowns = {}
    gutendex_failures = 0
    use_gutendex_random_pages = True

    try:
        while True:
            available_sources = [
                source for source in sources if cooldowns.get(source[0], 0) <= time.monotonic()
            ]
            if not available_sources:
                print("All sources are temporarily rate-limited; waiting before retrying.")
                time.sleep(delay)
                continue
            source_name, source_directory, crawl_source = available_sources[source_index % len(available_sources)]
            if source_name == "Gutendex":
                crawl_source = lambda: crawl_random_gutenberg(
                    max_download_bytes, seen_source_urls, use_gutendex_random_pages
                )
            try:
                print(f"[{work_number}] Fetching from {source_name}...")
                title, pages = crawl_source()
                print(f"  Downloaded from {source_name}: {title}")
                saved_path = save_work(source_directory, work_number, title, pages)
                print(f"  Staged {len(pages)} page(s): {saved_path.name}")
                if source_name == "Gutendex":
                    gutendex_failures = 0
                work_number += 1
                source_index += 1
            except HTTPError as error:
                if error.code == 429:
                    cooldowns[source_name] = time.monotonic() + rate_limit_cooldown
                    print(f"  {source_name} rate-limited; switching source immediately.")
                else:
                    print(f"  {source_name} failed with HTTP {error.code}; switching source.")
                source_index += 1
                continue
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
                if source_name == "Gutendex":
                    gutendex_failures += 1
                    if gutendex_failures >= 2 and use_gutendex_random_pages:
                        use_gutendex_random_pages = False
                        cooldowns.pop(source_name, None)
                        print(
                            "  Gutendex failed twice; falling back to first-page "
                            "selection for the rest of this run."
                        )
                    else:
                        cooldowns[source_name] = time.monotonic() + gutendex_failure_cooldown
                        print(
                            f"  Gutendex failed: {error}; cooling down for "
                            f"{gutendex_failure_cooldown:.0f}s before retrying."
                        )
                else:
                    print(f"  {source_name} failed: {error}; switching source.")
                source_index += 1
                continue
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nCrawler stopped.")

    if confirm("Pre-clean staged source text?"):
        preclean_staged_works(temp_directory)

    if not confirm("Call the Gemini API to filter staged files?"):
        print("Gemini filtering skipped. Staged files remain in temp_raw_text.")
        return

    if not api_key or not model_name:
        raise ValueError("Set GEMINI_API_KEY and GEMINI_MODEL in the project .env file before filtering")
    client = genai.Client(api_key=api_key)
    process_staged_works(
        temp_directory, output_directory, client, [model_name, *fallback_models], delay
    )


if __name__ == "__main__":
    main()