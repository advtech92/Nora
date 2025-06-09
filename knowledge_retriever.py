# knowledge_retriever.py

import os
import re
import requests
from bs4 import BeautifulSoup
import logging

# Where to dump new “.txt” files scraped from the web
SCRAPE_DIR = "data/books/scraped"

# Simple rate‐limiter to avoid hammering any one domain
import time
_last_request_time = {}


def fetch_url(url: str, min_interval: float = 1.0) -> str:
    """
    Fetch a URL’s HTML, enforcing at least `min_interval` seconds between requests
    to the same domain. Returns HTML string or empty string on failure.
    """
    from urllib.parse import urlparse

    domain = urlparse(url).netloc
    now = time.time()
    last = _last_request_time.get(domain, 0)
    wait = min_interval - (now - last)
    if wait > 0:
        time.sleep(wait)

    headers = {"User-Agent": "NoraScraper/1.0 (+https://your_project_url)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        _last_request_time[domain] = time.time()
        return response.text
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return ""


def clean_html(html: str) -> str:
    """
    Strip scripts, styles, and tags; return plain text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # remove scripts and styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # collapse multiple blank lines
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join([line for line in lines if line])
    return text


def save_text(content: str, title: str):
    """
    Save content to a UTF-8 .txt file under SCRAPE_DIR. Filename is derived from title.
    """
    os.makedirs(SCRAPE_DIR, exist_ok=True)
    # sanitize title → filename
    safe = re.sub(r"[^0-9a-zA-Z_\-]", "_", title)
    fname = f"{safe[:50]}.txt"
    path = os.path.join(SCRAPE_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"Saved scraped page to {path}")


def scrape_and_store(url: str):
    """
    High-level function: fetches URL, cleans HTML, extracts a title, and saves to a .txt.
    """
    html = fetch_url(url)
    if not html:
        return False

    text = clean_html(html)
    # extract <title> if present
    title = ""
    m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()
    else:
        title = url

    save_text(text, title)
    return True


# Example usage:
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python knowledge_retriever.py <url1> [<url2> ...]")
        sys.exit(1)

    for link in sys.argv[1:]:
        success = scrape_and_store(link)
        if success:
            print(f"Scraped: {link}")
        else:
            print(f"Failed to scrape: {link}")
