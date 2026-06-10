# rag-api/tests/test_report.py
import os
import report


def test_url_to_local_path_maps_reports_prefix(monkeypatch):
    monkeypatch.setattr(report.config, "SER_RENDER_URL", "http://render-api:8010")
    # A render-api URL like .../reports/<id>/<slug>.pdf maps under DATA_DIR.
    monkeypatch.setattr(report, "DATA_DIR", "/data")
    p = report._url_to_local_path("http://render-api:8010/reports/20260529-1/x.pdf")
    # Use os.path.join for cross-platform compatibility (runs in Linux containers).
    expected = os.path.join("/data", "reports", "20260529-1", "x.pdf")
    assert p == expected


def test_url_to_local_path_returns_none_for_foreign_url(monkeypatch):
    monkeypatch.setattr(report, "DATA_DIR", "/data")
    assert report._url_to_local_path("http://example.com/foo.pdf") is None


def test_url_to_local_path_blocks_traversal(monkeypatch):
    monkeypatch.setattr(report, "DATA_DIR", "/data")
    assert report._url_to_local_path("http://render-api:8010/reports/../../etc/passwd") is None


def test_closer_speaker_from_roles():
    # The speaker tagged 'closer' in the scorecard roles map.
    assert report._closer_speaker_from_roles(
        {"SPEAKER_00": "client", "SPEAKER_01": "closer"}) == "SPEAKER_01"
    # Falls back when roles is missing/empty/malformed.
    assert report._closer_speaker_from_roles(None) == "SPEAKER_01"
    assert report._closer_speaker_from_roles({}) == "SPEAKER_01"


def test_shape_utterance_renames_start_end_and_defaults():
    """Regression: render-api reads start/end (not start_sec/end_sec) and the
    transcript+charts come from this shape. Missing JSONB defaults to {}."""
    row = {
        "start_sec": 6.58, "end_sec": 7.1, "speaker": "SPEAKER_00",
        "text": "Hello.", "text_redacted": "Hello.", "pii_entities": None,
        "asr_confidence": 0.19, "is_backchannel": False, "question_types": None,
        "sentiment_label": "positive", "sentiment_score": 0.55,
        "emotion_label": "happy", "emotion_probs": {"happy": 0.9},
        "valence": 0.99, "arousal": 0.99, "dominance": 0.89,
    }
    u = report._shape_utterance(row, "en")
    assert u["start"] == 6.58 and u["end"] == 7.1   # renamed from *_sec
    assert "start_sec" not in u and "end_sec" not in u
    assert u["text"] == "Hello." and u["speaker"] == "SPEAKER_00"
    assert u["emotion_probs"] == {"happy": 0.9}
    assert u["question_types"] == {} and u["pii_entities"] == {}  # None -> {}
    assert u["language"] == "en"
