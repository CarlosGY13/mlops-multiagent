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
    # Offline/local fallback compatible with the UI contract.
    papers = [
        {
            "title": f"Related study {i+1} on {query}",
            "source": "Europe PMC (mock)",
            "year": "",
            "venue": "",
            "citations": "",
            "open_access": "",
            "snippet": "Mock result (offline mode).",
            "url": f"https://europepmc.org/",
        }
        for i in range(top_k)
    ]
    datasets = [
        {
            "title": f"Example dataset {i+1} for {query}",
            "source": "OpenML (mock)",
            "url": "https://www.openml.org/",
        }
        for i in range(min(2, top_k))
    ]
    return RagResult(papers=papers, datasets=datasets)


# Note: no local cache by design (per product decision). Keep calls live and lightweight.


def _compact_snippet(text: str, max_len: int = 220) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return ""
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _europe_pmc_url(doi: str, pmid: str, pmcid: str) -> str:
    doi = (doi or "").strip()
    if doi:
        if doi.lower().startswith("http"):
            return doi
        return f"https://doi.org/{doi}"

    pmcid = (pmcid or "").strip()
    if pmcid:
        pmc = pmcid.replace("PMC", "").strip()
        return f"https://europepmc.org/article/PMC/{pmc}"

    pmid = (pmid or "").strip()
    if pmid:
        return f"https://europepmc.org/article/MED/{pmid}"

    return "https://europepmc.org/"


def _search_europe_pmc(query: str, top_k: int, client: httpx.Client) -> List[Dict[str, str]]:
    # Europe PMC REST API
    # Docs: https://europepmc.org/RestfulWebService
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "pageSize": str(top_k),
        "sort": "CITED desc",
    }
    r = client.get(url, params=params)
    r.raise_for_status()
    js = r.json()

    results = ((js.get("resultList") or {}).get("result")) or []
    papers: List[Dict[str, str]] = []
    for x in results[:top_k]:
        title = (x.get("title") or "").strip() or "Untitled paper"
        year = str(x.get("pubYear") or "").strip()
        venue = (x.get("journalTitle") or x.get("bookOrReportDetails") or "").strip()
        doi = (x.get("doi") or "").strip()
        pmid = str(x.get("pmid") or "").strip()
        pmcid = str(x.get("pmcid") or "").strip()
        citations = str(x.get("citedByCount") or "").strip()
        oa = str(x.get("isOpenAccess") or "").strip()
        abs_txt = (x.get("abstractText") or "").strip()

        url_item = _europe_pmc_url(doi=doi, pmid=pmid, pmcid=pmcid)

        papers.append(
            {
                "title": title,
                "source": "Europe PMC",
                "year": year,
                "venue": venue,
                "citations": citations,
                "open_access": oa,
                "snippet": _compact_snippet(abs_txt),
                "url": url_item,
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
            papers = _search_europe_pmc(query, top_k, client)
            # simple dataset search via OpenML tokens
            datasets = _search_openml(query, top_k=min(5, top_k), client=client)
    except Exception:
        # Fail closed into mock to keep UX stable.
        return _mock_results(query, top_k)

    # Ensure contract: always return exactly top_k paper items (pad with mock if needed)
    if len(papers) < top_k:
        papers = papers + _mock_results(query, top_k).papers[len(papers) : top_k]

    return RagResult(papers=papers[:top_k], datasets=datasets)
