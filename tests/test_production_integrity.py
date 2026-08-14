from app.services.ai.interview_agent import normalize_text, jaccard_similarity
from app.services.research.evidence import classify_source_type

def test_jaccard_similarity_deduplication():
    q1 = "Explain B-Tree vs Hash indexing in PostgreSQL and when you would select each for a high-concurrency service."
    q2 = "Explain B-Tree vs Hash indexing in PostgreSQL and when you would select each for high concurrency."
    q3 = "How do you optimize React state management using Context API?"

    sim1_2 = jaccard_similarity(q1, q2)
    sim1_3 = jaccard_similarity(q1, q3)

    assert sim1_2 > 0.70, "Near-duplicate questions should be identified with high similarity score."
    assert sim1_3 < 0.20, "Distinct technical questions should have low similarity score."

def test_source_integrity_classification():
    url_official = "https://careers.tcs.com/jobs/software-engineer"
    url_public = "https://www.geeksforgeeks.org/tcs-interview-experience/"
    url_unknown = "https://unknownblog.io/post"

    assert classify_source_type(url_official, "TCS") == "Official 🟢"
    assert classify_source_type(url_public, "TCS") == "Publicly Reported 🔵"
    assert classify_source_type(url_unknown, "TCS") == "Publicly Reported 🔵"
