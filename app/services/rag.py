from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import httpx


@dataclass
class RagResult:
    papers: List[Dict[str, str]]
    datasets: List[Dict[str, str]]


def _mock_results(query: str, top_k: int) -> RagResult:
    # Offline/local fallback compatible with the OpenAlex-focused contract.
    papers = [
        {
            "title": f"Related study {i+1} on {query}",
            "source": "OpenAlex",
            "url": f"https://example.org/papers/{i+1}",
        }
        for i in range(top_k)
    ]
    datasets: List[Dict[str, str]] = []
    return RagResult(papers=papers, datasets=datasets)


# Note: no local cache by design (per product decision). Keep calls live and lightweight.


def _reconstruct_abstract(inv: Dict[str, List[int]]) -> str:
    if not inv:
        return ""
    try:
        words: List[str] = []
        for word, positions in inv.items():
            for pos in positions:
                if len(words) <= pos:
                    words.extend([""] * (pos - len(words) + 1))
                words[pos] = word
        return " ".join([w for w in words if w]).strip()
    except Exception:
        return ""


def _search_openalex(query: str, top_k: int, client: httpx.Client) -> List[Dict[str, str]]:
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "sort": "cited_by_count:desc",
        "per_page": str(top_k),
        "filter": "type:article",
        "select": "id,doi,title,publication_year,cited_by_count,open_access,primary_location,abstract_inverted_index",
        "mailto": "biopapers@app.local",
    }
    r = client.get(url, params=params)
    r.raise_for_status()
    js = r.json()

    results = js.get("results") or []
    papers: List[Dict[str, str]] = []
    for x in results[:top_k]:
        title = (x.get("title") or "").strip() or "Untitled paper"
        year = str(x.get("publication_year") or "").strip()
        journal = (((x.get("primary_location") or {}).get("source") or {}).get("display_name") or "").strip()
        doi = (x.get("doi") or "").strip()
        doi_url = doi or (x.get("id") or "").strip()
        citations = x.get("cited_by_count") or 0
        oa = ((x.get("open_access") or {}).get("is_oa"))
        abstract = _reconstruct_abstract((x.get("abstract_inverted_index") or {}))

        extras: List[str] = []
        if journal:
            extras.append(journal)
        if year:
            extras.append(year)
        suffix = f" ({', '.join(extras)})" if extras else ""
        oa_tag = " · OA" if oa else ""
        cite_tag = f" · {citations} citations" if citations else ""
        abstract_preview = f" · {abstract[:160]}..." if abstract else ""
        papers.append(
            {
                "title": f"{title}{suffix}{oa_tag}{cite_tag}{abstract_preview}",
                "source": "OpenAlex",
                "url": doi_url,
            }
        )
    return papers


def _tokens(query: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (query or "").lower())
    # keep meaningful tokens only
    stop = {"and", "or", "the", "a", "an", "of", "in", "on", "for", "to", "with", "from"}
    toks = [t for t in toks if len(t) >= 3 and t not in stop]
    return toks[:6]


def _openml_list_data_name(name: str, top_k: int, client: httpx.Client) -> List[Dict[str, str]]:
    # OpenML uses path-based params; query params are ignored.
    safe = httpx.URL(name).raw_path.decode('utf-8')
    url = f"https://www.openml.org/api/v1/json/data/list/data_name/{safe}/limit/{top_k}"
    r = client.get(url)
    if r.status_code == 412:
        return []
    r.raise_for_status()
    js = r.json()
    ds = (js.get("data") or {}).get("dataset") or []
    out: List[Dict[str, str]] = []
    for d in ds[:top_k]:
        did = str(d.get("did") or "").strip()
        title = (d.get("name") or "").strip() or f"OpenML dataset {did}"
        url_item = f"https://www.openml.org/d/{did}" if did else "https://www.openml.org/search?type=data"
        out.append({"title": title, "source": "OpenML", "url": url_item})
    return out


def _openml_list_tag(tag: str, top_k: int, client: httpx.Client) -> List[Dict[str, str]]:
    safe = httpx.URL(tag).raw_path.decode('utf-8')
    url = f"https://www.openml.org/api/v1/json/data/list/tag/{safe}/limit/{top_k}"
    r = client.get(url)
    if r.status_code == 412:
        return []
    r.raise_for_status()
    js = r.json()
    ds = (js.get("data") or {}).get("dataset") or []
    out: List[Dict[str, str]] = []
    for d in ds[:top_k]:
        did = str(d.get("did") or "").strip()
        title = (d.get("name") or "").strip() or f"OpenML dataset {did}"
        url_item = f"https://www.openml.org/d/{did}" if did else "https://www.openml.org/search?type=data"
        out.append({"title": title, "source": "OpenML", "url": url_item})
    return out


def _search_openml(query: str, top_k: int, client: httpx.Client) -> List[Dict[str, str]]:
    # Kept for compatibility; currently not used by Agent side panel request.
    toks = _tokens(query)
    candidates = []
    if query:
        candidates.append(query.strip().lower().replace(" ", "_"))
    candidates.extend(toks)

    seen = set()
    for c in candidates:
        c = c.strip("_-")
        if not c or c in seen:
            continue
        seen.add(c)
        hits = _openml_list_data_name(c, top_k, client)
        if hits:
            return hits
    return []


def search_scientific_context(query: str, top_k: int = 5, use_local_mock: bool = True) -> RagResult:
    if use_local_mock:
        return _mock_results(query, top_k)

    try:
        with httpx.Client(timeout=20, headers={"User-Agent": "LabNotebookAI/1.1"}) as client:
            papers = _search_openalex(query, top_k, client)
            datasets: List[Dict[str, str]] = []
    except Exception:
        # Fail closed into mock to keep UX stable.
        return _mock_results(query, top_k)

    # Ensure contract: always return exactly top_k items (pad with mock if needed)
    if len(papers) < top_k:
        papers = papers + _mock_results(query, top_k).papers[len(papers) : top_k]

    return RagResult(papers=papers[:top_k], datasets=datasets)
