"""Basic tests for clawbot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clawbot.models import Paper
from clawbot.dedup import DedupEngine
from clawbot.filter import is_llm_serving


def test_paper_fingerprint():
    p1 = Paper(title="Test Paper", arxiv_id="2401.12345")
    p2 = Paper(title="Test Paper", arxiv_id="2401.12345")
    p3 = Paper(title="Different Paper", arxiv_id="2401.99999")

    assert p1.fingerprint() == p2.fingerprint()
    assert p1.fingerprint() != p3.fingerprint()
    print("✅ test_paper_fingerprint passed")


def test_dedup():
    existing = [Paper(title="Existing", arxiv_id="2401.00001")]
    dedup = DedupEngine(existing)

    assert not dedup.is_new(Paper(title="Existing", arxiv_id="2401.00001"))
    assert dedup.is_new(Paper(title="New", arxiv_id="2401.00002"))
    print("✅ test_dedup passed")


def test_filter_positive():
    p = Paper(
        title="Fast LLM Serving with Speculative Decoding",
        abstract_en="We propose a new method for accelerating LLM inference using speculative decoding and KV cache optimization."
    )
    assert is_llm_serving(p)
    print("✅ test_filter_positive passed")


def test_filter_negative():
    p = Paper(
        title="Fine-tuning LLMs for Medical Diagnosis",
        abstract_en="We fine-tune large language models for clinical decision support and drug discovery."
    )
    assert not is_llm_serving(p)
    print("✅ test_filter_negative passed")


def test_legacy_migration():
    legacy = {
        "title": "Old Paper",
        "summary_en": "This is the abstract",
        "github": "user/repo",
        "venue": "NeurIPS",
        "year": 2024,
        "id": 42,
    }
    p = Paper.from_legacy(legacy)
    assert p.abstract_en == "This is the abstract"
    assert p.github_repo == "user/repo"
    assert p.conference == "NeurIPS"
    assert p.year == 2024
    assert p.id == 42
    print("✅ test_legacy_migration passed")


def run_all():
    test_paper_fingerprint()
    test_dedup()
    test_filter_positive()
    test_filter_negative()
    test_legacy_migration()
    print("\n🎉 All tests passed")


if __name__ == "__main__":
    run_all()
