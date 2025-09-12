# crawl_light.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List, Set

def crawl_site(start_url: str, max_pages: int = 10, ua: str = "AseonBot/0.1") -> Dict[str, Any]:
    """
    Very lightweight crawler for MVP:
    - Follows same-domain links up to max_pages (BFS-ish).
    - Captures: status, title, meta description, H1–H3, canonical.
    - Returns a compact dict consumed by the schema agent as context.
    """
    visited: Set[str] = set()
    to_visit: List[str] = [start_url]
    pages: List[Dict[str, Any]] = []
    count = 0

    headers = {"User-Agent": ua}
    base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(start_url))

    while to_visit and count < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        count += 1

        status = None
        html = ""
        try:
            r = requests.get(url, headers=headers, timeout=10)
            status = r.status_code
            # follow simple 30x by recording the page with status only (no HTML parse)
            if status == 200:
                html = r.text
        except Exception as e:
            pages.append({"url": url, "status": "error", "error": str(e)})
            continue

        title = None
        description = None
        h1 = None
        h2: List[str] = []
        h3: List[str] = []
        canonical = None

        soup = BeautifulSoup(html, "html.parser") if html else None
        if soup:
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag and desc_tag.get("content"):
                description = desc_tag["content"].strip()

            h1_tag = soup.find("h1")
            if h1_tag:
                h1 = h1_tag.get_text(strip=True)

            for tag in soup.find_all("h2"):
                h2.append(tag.get_text(strip=True))
            for tag in soup.find_all("h3"):
                h3.append(tag.get_text(strip=True))

            link_tag = soup.find("link", rel="canonical")
            if link_tag and link_tag.get("href"):
                canonical = link_tag["href"]

            # discover internal links for further crawl
            for a in soup.find_all("a", href=True):
                abs_url = urljoin(base, a["href"])
                if abs_url.startswith(base) and abs_url not in visited:
                    to_visit.append(abs_url)

        pages.append({
            "url": url,
            "status": status if status is not None else "error",
            "title": title,
            "meta_description": description,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "canonical": canonical,
        })

    summary = {
        "pages_total": len(pages),
        "ok_200": sum(1 for p in pages if isinstance(p.get("status"), int) and p["status"] == 200),
        "errors": sum(1 for p in pages if p.get("status") == "error"),
    }

    return {
        "start_url": start_url,
        "pages": pages,
        "summary": summary
    }
