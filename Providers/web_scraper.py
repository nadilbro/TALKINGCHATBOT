"""
website_scraper.py

Scrapes a business's website and stores the content as document chunks in the
existing RAG pipeline. Called from the dashboard when a customer sets up their
API key, or on-demand via a "refresh" endpoint.

Dependencies to add to requirements.txt:
    httpx>=0.25.0
    trafilatura>=1.6.0
    beautifulsoup4>=4.12.0
"""

import asyncio
import httpx
import traceback
import uuid
from urllib.parse import urljoin, urlparse
from typing import Optional
import trafilatura
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = "Chatabit/1.0 (+https://chatabit.ai/bot)"
MAX_PAGES = 20
PAGE_TIMEOUT_SECONDS = 10
DELAY_BETWEEN_PAGES_SECONDS = 1.0
MAX_CONTENT_LENGTH = 500_000  # 500KB per page, anything bigger is probably not text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def scrape_and_ingest_website(
    api_key: str,
    website_url: str,
    rag,
    replace_existing: bool = True,
) -> dict:
    """
    Fetch a business's website, extract clean text from each page, chunk it,
    embed it, and store the chunks in the RAG documents table keyed by api_key.

    Args:
        api_key: The customer's API key — used as the document owner.
        website_url: The root URL to scrape, e.g. "https://bubbleworks.com.au"
        rag: Your VectorRAGService instance.
        replace_existing: If True, deletes any existing scraped documents for
            this api_key before adding new ones. Set to False for incremental
            additions.

    Returns:
        A dict with the scraping results — page counts, errors, chunk counts.
    """
    result = {
        "website_url": website_url,
        "pages_fetched": 0,
        "pages_failed": 0,
        "chunks_created": 0,
        "errors": [],
        "success": False,
    }

    try:
        # Normalize the URL
        if not website_url.startswith(("http://", "https://")):
            website_url = "https://" + website_url
        
        parsed_root = urlparse(website_url)
        root_domain = parsed_root.netloc.lower()
        if not root_domain:
            result["errors"].append("Invalid URL — no domain found")
            return result

        # If replacing, wipe any existing scraped documents for this key
        if replace_existing:
            try:
                rag.deleteScrapedDocuments(api_key)
            except Exception as e:
                print(f"==> Could not delete old scraped docs (may not exist yet): {e}")

        # Fetch the site
        pages = await _fetch_site(website_url, root_domain)
        result["pages_fetched"] = len(pages)

        if not pages:
            result["errors"].append("No pages could be fetched")
            return result

        # Create a single doc_id for this scrape batch so they can be managed together
        doc_id = f"website_{uuid.uuid4().hex[:12]}"

        total_chunks = 0
        for page_url, page_text in pages.items():
            if not page_text or len(page_text.strip()) < 50:
                continue

            # Chunk the page content
            chunks = _chunk_text(page_text, chunk_size=500, overlap=50)

            for i, chunk_content in enumerate(chunks):
                # Prefix each chunk with its source URL so retrieval context
                # includes attribution
                labeled_chunk = f"[From {page_url}]\n\n{chunk_content}"

                try:
                    embedding = await rag.embedText(labeled_chunk)
                    rag.storeDocumentChunk(
                        doc_id=doc_id,
                        api_key=api_key,
                        chunk_index=total_chunks,
                        content=labeled_chunk,
                        embedding=embedding,
                        filename=f"website:{root_domain}",
                    )
                    total_chunks += 1
                except Exception as e:
                    print(f"==> Failed to store chunk from {page_url}: {e}")
                    result["errors"].append(f"Chunk store failed for {page_url}")

        result["chunks_created"] = total_chunks
        result["success"] = total_chunks > 0

        print(f"==> Scrape complete for {website_url}: {result['pages_fetched']} pages, {total_chunks} chunks")

    except Exception as e:
        traceback.print_exc()
        result["errors"].append(f"Scrape failed: {str(e)}")

    return result


# ---------------------------------------------------------------------------
# Fetching logic
# ---------------------------------------------------------------------------

async def _fetch_site(root_url: str, root_domain: str) -> dict:
    """
    Fetches the root URL and up to MAX_PAGES internal links, returning a dict
    of {url: clean_text_content}.
    """
    pages = {}
    visited = set()
    to_visit = [root_url]

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=PAGE_TIMEOUT_SECONDS,
        follow_redirects=True,
    ) as client:
        while to_visit and len(pages) < MAX_PAGES:
            url = to_visit.pop(0)
            normalized = _normalize_url(url)
            
            if normalized in visited:
                continue
            visited.add(normalized)

            try:
                response = await client.get(url)
                if response.status_code != 200:
                    print(f"==> Skipping {url}: status {response.status_code}")
                    continue

                content_type = response.headers.get("content-type", "").lower()
                if "html" not in content_type:
                    continue

                html = response.text
                if len(html) > MAX_CONTENT_LENGTH:
                    html = html[:MAX_CONTENT_LENGTH]

                # Extract clean text using trafilatura (purpose-built for this)
                clean_text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    include_links=False,
                    favor_precision=True,
                )

                if clean_text and len(clean_text.strip()) >= 50:
                    pages[url] = clean_text
                    print(f"==> Fetched {url}: {len(clean_text)} chars")

                # Extract internal links from this page to crawl next
                if len(pages) < MAX_PAGES:
                    new_links = _extract_internal_links(html, url, root_domain)
                    for link in new_links:
                        if _normalize_url(link) not in visited:
                            to_visit.append(link)

                # Be polite — wait between requests
                await asyncio.sleep(DELAY_BETWEEN_PAGES_SECONDS)

            except httpx.TimeoutException:
                print(f"==> Timeout fetching {url}")
            except Exception as e:
                print(f"==> Failed to fetch {url}: {e}")

    return pages


def _extract_internal_links(html: str, current_url: str, root_domain: str) -> list:
    """
    Extracts internal links from an HTML page. Only returns URLs that are on
    the same domain as the root.
    """
    links = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            
            # Skip anchors, mailto, tel, javascript
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            
            # Resolve relative URLs against the current page
            absolute_url = urljoin(current_url, href)
            parsed = urlparse(absolute_url)
            
            # Only keep links on the same domain
            if parsed.netloc.lower() != root_domain:
                continue
            
            # Strip fragments and query strings for deduplication
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            # Skip common non-content URLs
            skip_patterns = [
                "/wp-admin", "/wp-login", "/wp-content/uploads",
                "/feed/", "/rss/", "/sitemap",
                ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
                ".mp4", ".mp3", ".zip", ".doc", ".docx",
            ]
            if any(pattern in clean_url.lower() for pattern in skip_patterns):
                continue
            
            links.append(clean_url)
    except Exception as e:
        print(f"==> Link extraction failed: {e}")
    
    return links


def _normalize_url(url: str) -> str:
    """Normalizes a URL for deduplication — strips fragments, trailing slashes, lowercases domain."""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text into overlapping chunks for embedding. Word-based chunking,
    same approach as your existing document uploader.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks