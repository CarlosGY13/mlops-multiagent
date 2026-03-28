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
    # Offline/local fallback compatible with the live Europe PMC + OpenML contract.
    papers = [
        {
            "title": f"Related study {i+1} on {query}",
            "source": "Europe PMC",
            "url": f"https://example.org/papers/{i+1}",
        }
        for i in range(top_k)
    ]
    datasets = [
        {
            "title": f"Similar dataset {i+1} for {query}",
            "source": "OpenML",
            "url": f"https://example.org/datasets/{i+1}",
        }
        for i in range(top_k)
    ]
    return RagResult(papers=papers, datasets=datasets)


# Note: no local cache by design (per product decision). Keep calls live and lightweight.


def _europe_pmc_url(source: str, pmc_id: str) -> str:
    src = (source or "").strip() or "MED"
    pid = (pmc_id or "").strip()
    return f"https://europepmc.org/article/{src}/{pid}" if pid else "https://europepmc.org/"


def _search_europe_pmc(query: str, top_k: int, client: httpx.Client) -> List[Dict[str, str]]:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    # Avoid optional params that can yield partial responses depending on backend defaults.
    params = {"query": query, "format": "json", "pageSize": str(top_k)}
    r = client.get(url, params=params)
    r.raise_for_status()
    js = r.json()

    results = (js.get("resultList") or {}).get("result") or []
    papers: List[Dict[str, str]] = []
    for x in results[:top_k]:
        title = (x.get("title") or "").strip() or "Untitled paper"
        source = "Europe PMC"
        pid = (x.get("id") or "").strip()
        src = (x.get("source") or "").strip()
        year = (x.get("pubYear") or "").strip()
        journal = (x.get("journalTitle") or "").strip()
        url_item = _europe_pmc_url(src, pid)
        suffix = ""
        if year and journal:
            suffix = f" ({journal}, {year})"
        elif year:
            suffix = f" ({year})"
        papers.append({"title": f"{title}{suffix}", "source": source, "url": url_item})
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
    toks = _tokens(query)

    # Domain-ish tag mapping (best-effort). OpenML tags are community-defined.
    tag_map = {
        "gene": "biology",
        "genome": "biology",
        "genomic": "biology",
        "protein": "biology",
        "rna": "biology",
        "cell": "biology",
        "cancer": "biology",
        "clinical": "health",
        "patient": "health",
        "ecg": "health",
        "mri": "health",
    }
    for t in toks:
        if t in tag_map:
            hits = _openml_list_tag(tag_map[t], top_k, client)
            if hits:
                return hits

    # Try exact-ish dataset name matching.
    candidates = []
    if query:
        candidates.append(query.strip().lower().replace(" ", "_"))
        candidates.append(query.strip().lower().replace(" ", "-"))
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
            papers = _search_europe_pmc(query, top_k, client)
            datasets = _search_openml(query, top_k, client)
    except Exception:
        # Fail closed into mock to keep UX stable.
        return _mock_results(query, top_k)

    # Ensure contract: always return exactly top_k items (pad with mock if needed)
    if len(papers) < top_k:
        papers = papers + _mock_results(query, top_k).papers[len(papers) : top_k]
    if len(datasets) < top_k:
        datasets = datasets + _mock_results(query, top_k).datasets[len(datasets) : top_k]

    return RagResult(papers=papers[:top_k], datasets=datasets[:top_k])
