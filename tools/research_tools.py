import os
from collections import Counter

import requests
from crewai.tools import tool

OPENALEX = "https://api.openalex.org"
CROSSREF = "https://api.crossref.org"


def _contact() -> dict:
    """Optional: set OPENALEX_EMAIL in your environment for faster, 'polite' API access."""
    email = os.getenv("OPENALEX_EMAIL")
    return {"mailto": email} if email else {}


def _openalex_works(query: str, per_page: int, pakistan_only: bool, select: str = ""):
    params = {"search": query, "per-page": per_page, **_contact()}
    if pakistan_only:
        # Works with at least one author affiliated to a Pakistani institution
        params["filter"] = "authorships.institutions.country_code:PK"
    if select:
        params["select"] = select
    response = requests.get(f"{OPENALEX}/works", params=params, timeout=15)
    response.raise_for_status()
    return response.json().get("results", [])


def _format_works(results) -> str:
    lines = []
    for item in results:
        title = item.get("title") or "Untitled work"
        doi = item.get("doi") or "No DOI listed"
        year = item.get("publication_year") or "Year not listed"
        lines.append(
            f"- **{title}** ({year}) | DOI: {doi} | "
            f"OpenAlex record: {item.get('id', 'not provided')}"
        )
    return "\n".join(lines)


@tool("Search Global Scholarly Literature (OpenAlex RAG)")
def search_global_literature(query: str) -> str:
    """Search OpenAlex for worldwide scholarly works on a legal research topic.

    OpenAlex metadata is a discovery aid, not confirmation that a work is peer-reviewed
    or that its citation details are authoritative.
    """
    try:
        results = _openalex_works(query, 5, pakistan_only=False)
    except (requests.RequestException, ValueError) as error:
        return f"OpenAlex search failed: {error}"
    return _format_works(results) or "OpenAlex returned no records for this query."


@tool("Search Pakistan-Affiliated Scholarship (OpenAlex)")
def search_pakistan_literature(query: str) -> str:
    """Search OpenAlex for works with at least one author at a Pakistani institution.

    This finds scholarship by Pakistan-based authors, which is not always the same as
    scholarship about Pakistani law. Say so in the answer.
    """
    try:
        results = _openalex_works(query, 5, pakistan_only=True)
    except (requests.RequestException, ValueError) as error:
        return f"OpenAlex (Pakistan filter) search failed: {error}"
    return _format_works(results) or "No Pakistan-affiliated records found for this query."


@tool("Pakistan Code Official-Source Verification Guide")
def search_pakistan_code(query: str) -> str:
    """Direct the researcher to official Pakistani primary sources for verification.

    Automated search of these portals is not connected. Do not present suggested acts,
    provisions or cases as retrieved results.
    """
    return (
        "Automated search of Pakistani primary law is not configured, so no statutes or "
        "cases were retrieved. Check the official sources directly and confirm the current "
        f"text and status (research topic: {query}):\n"
        "- Statutes: https://pakistancode.gov.pk/\n"
        "- Supreme Court judgments: https://www.supremecourt.gov.pk/\n"
        "- National Assembly (bills and acts): https://na.gov.pk/\n"
        "- High Court websites (e.g. Lahore High Court: https://lhc.gov.pk/)"
    )


@tool("Verify Reference with Crossref")
def verify_reference_crossref(reference_text: str) -> str:
    """Look up a journal article or book reference in Crossref to confirm it exists.

    Crossref only covers works with DOIs. A missing result does not prove a reference is
    false, and law reports such as PLD or SCMR will not appear here.
    """
    params = {"query.bibliographic": reference_text, "rows": 3, **_contact()}
    try:
        response = requests.get(f"{CROSSREF}/works", params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
    except (requests.RequestException, ValueError) as error:
        return f"Crossref lookup failed: {error}"

    lines = []
    for item in items:
        title = (item.get("title") or ["Untitled"])[0]
        journal = (item.get("container-title") or ["(no container listed)"])[0]
        authors = ", ".join(
            f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
            for a in item.get("author", [])[:4]
        ) or "authors not listed"
        year = (item.get("issued", {}).get("date-parts") or [[None]])[0][0]
        lines.append(
            f"- {authors} | **{title}** | {journal} | {year} | vol {item.get('volume', '-')}"
            f", issue {item.get('issue', '-')}, pages {item.get('page', '-')} | "
            f"DOI: {item.get('DOI', 'none')}"
        )
    return "\n".join(lines) if lines else "Crossref found no matching records."


@tool("Find Journals Publishing Similar Work (OpenAlex)")
def find_journals(topic: str, region: str = "world") -> str:
    """Show which journals have published articles on a topic, using OpenAlex.

    Use region='pakistan' to count only articles with Pakistan-affiliated authors, or
    region='world' for all. This is evidence of fit, not of indexing or HEC/HJRS status.
    """
    pakistan_only = region.strip().lower() == "pakistan"
    try:
        results = _openalex_works(
            topic, 50, pakistan_only, select="id,primary_location"
        )
    except (requests.RequestException, ValueError) as error:
        return f"OpenAlex journal search failed: {error}"

    counts, info = Counter(), {}
    for item in results:
        source = (item.get("primary_location") or {}).get("source") or {}
        name = source.get("display_name")
        if not name or source.get("type") not in (None, "journal"):
            continue
        counts[name] += 1
        info[name] = source
    if not counts:
        return "No journal information found for this topic."

    lines = []
    for name, n in counts.most_common(8):
        src = info[name]
        lines.append(
            f"- **{name}** | {n} similar articles in sample | "
            f"in DOAJ (open access list): {src.get('is_in_doaj')} | ISSN-L: {src.get('issn_l') or 'n/a'}"
        )
    note = (
        "\nVerify each journal's scope, indexing (Scopus/HEC HJRS), fees and "
        "double-blind policy on its own website before submitting."
    )
    return "\n".join(lines) + note
