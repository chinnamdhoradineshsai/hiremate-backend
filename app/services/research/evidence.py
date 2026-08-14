from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

@dataclass
class EvidenceRecord:
    title: str
    url: str
    snippet: str
    domain: str
    source_type: str
    content: Optional[str] = ""

def classify_source_type(url: str, company_name: str) -> str:
    """
    Strict Source Integrity Classifier.
    - Official 🟢: ONLY when URL domain matches company's official domain.
    - Publicly Reported 🔵: ONLY when URL belongs to candidate/interview experience portals.
    - AI Generated 🟣: Reserved for AI-inferred insights and synthetic scenarios.
    """
    if not url or not url.startswith("http"):
        return "AI Generated 🟣"

    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    company_clean = company_name.lower().replace(" ", "").replace(".", "")

    # Official company domain check
    if company_clean and company_clean in domain:
        return "Official 🟢"

    # Public candidate interview experience portals
    candidate_domains = [
        "geeksforgeeks.org", "glassdoor.com", "leetcode.com", "ambitionbox.com",
        "indeed.com", "blind.com", "reddit.com", "interviewbit.com", "medium.com",
        "hackerrank.com", "github.com", "indiabix.com", "careerbliss.com"
    ]
    if any(cd in domain for cd in candidate_domains):
        return "Publicly Reported 🔵"

    return "Publicly Reported 🔵"
