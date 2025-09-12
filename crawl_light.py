# crawl_light.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def crawl_site(start_url: str, max_pages: int = 10, ua: str = "AseonBot/0.1") -> dict:
    """
    Very lightweight crawler: follows same-domain links up to max_pages.
    Collects title, meta description, H1–H3 for each page.
    """
    visited = set()
    to_visit = [start_url]
    pages = []
    count = 0

    headers = {"User-Agent": ua}

    while to_visit and count < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        count += 1

        try:
            r = requests.get(url, headers=headers, timeout=10)
            status = r.status_code
            html = r.text if status == 200 else ""
        except Exception as e:
            pages.append({"url": url, "status": "error", "error": str(e)})
            continue

        soup = BeautifulSoup(html, "html.parser") if html else None
        title = soup.title.string.strip() if soup and soup.title else None
        description = None
        h1 = None
        h2 = []
        h3 = []
        canonical = None

        if soup:
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag and desc_tag.get("content"):
                description = desc_tag["content"]

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

            # Voeg interne links toe om verder te crawlen
            base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(start_url))
            for a in soup.find_all("a", href=True):
                abs_url = urljoin(base, a["href"])
                if abs_url.startswith(base) and abs_url not in visited:
                    to_visit.append(abs_url)

        pages.append({
            "url": url,
            "status": status,
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

    return {"start_url": start_url, "pages": pages, "summary": summary}
