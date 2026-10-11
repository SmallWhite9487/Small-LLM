"""Download Wikisource works and their detected chapters for local NLP training.

Examples:
    python wikisource_crawler.py 5
    python wikisource_crawler.py --titles "Pride and Prejudice" "The Raven"
"""

import argparse
import html
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen


API_URL = "https://en.wikisource.org/w/api.php"
WIKI_URL = "https://en.wikisource.org/wiki/"
USER_AGENT = "EducationalWikisourceCrawler/1.0 (local text-generation project)"
CHAPTER_PATTERN = re.compile(r"\b(chapter|book|part|act|scene|canto|volume)\b", re.IGNORECASE)


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
        if tag == "a" and self._ignored_depth == 0:
            self._current_href = attributes.get("href")
            self._current_link_text = []

    def handle_endtag(self, tag):
        if tag == "a" and self._current_href is not None:
            self.links.append((self._current_href, "".join(self._current_link_text).strip()))
            self._current_href = None
        if tag in {"script", "style", "table", "sup"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data):
        if self._ignored_depth == 0:
            self.text_parts.append(data)
            if self._current_href is not None:
                self._current_link_text.append(data)


def api_request(parameters):
    parameters = {"format": "json", "formatversion": "2", **parameters}
    query = "&".join(f"{quote(str(key))}={quote(str(value))}" for key, value in parameters.items())
    request = Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def random_titles(count):
    response = api_request(
        {"action": "query", "list": "random", "rnnamespace": 0, "rnlimit": count}
    )
    return [page["title"] for page in response["query"]["random"]]


def fetch_page(title):
    response = api_request({"action": "parse", "page": title, "prop": "text"})
    parsed = PageHTMLParser()
    parsed.feed(response["parse"]["text"])
    text = re.sub(r"\s+", " ", html.unescape(" ".join(parsed.text_parts))).strip()
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


def crawl_work(title, delay, max_chapters):
    work_title, work_text, links = fetch_page(title)
    pages = [(work_title, work_text)]
    seen_titles = {work_title}

    for chapter_title in chapter_titles(work_title, links):
        if len(pages) >= max_chapters + 1 or chapter_title in seen_titles:
            break
        time.sleep(delay)
        try:
            chapter_title, chapter_text, _ = fetch_page(chapter_title)
        except (OSError, KeyError) as error:
            print(f"  Skipping chapter {chapter_title!r}: {error}")
            continue
        seen_titles.add(chapter_title)
        pages.append((chapter_title, chapter_text))

    return pages


def save_work(output_directory, number, requested_title, pages):
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", requested_title).strip("_")[:60]
    output_path = output_directory / f"{number:03d}_{safe_name}.txt"
    sections = []
    for page_title, page_text in pages:
        source_url = WIKI_URL + quote(page_title.replace(" ", "_"))
        sections.append(f"Title: {page_title}\nSource: {source_url}\n\n{page_text}")
    output_path.write_text("\n\n".join(sections), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download Wikisource works and detected chapters.")
    parser.add_argument("count", nargs="?", type=int, help="Number of random works to download.")
    parser.add_argument("--titles", nargs="+", help="Specific Wikisource work titles to download.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "raw_text")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API requests.")
    parser.add_argument("--max-chapters", type=int, default=200)
    args = parser.parse_args()

    if args.count is None and not args.titles:
        args.count = int(input("Number of Wikisource works to download: "))
    if args.count is not None and args.count < 1:
        parser.error("count must be at least 1")

    titles = args.titles or random_titles(args.count)
    args.output.mkdir(parents=True, exist_ok=True)

    for number, title in enumerate(titles, start=1):
        print(f"[{number}/{len(titles)}] Downloading {title}")
        try:
            pages = crawl_work(title, args.delay, args.max_chapters)
        except (OSError, KeyError) as error:
            print(f"  Failed: {error}")
            continue
        saved_path = save_work(args.output, number, title, pages)
        print(f"  Saved {len(pages)} page(s) to {saved_path}")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()