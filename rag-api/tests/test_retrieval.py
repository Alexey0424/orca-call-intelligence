# rag-api/tests/test_retrieval.py
import retrieval


def test_vec_literal_formats_pgvector_array():
    assert retrieval._vec_literal([0.5, -1.0, 0.0]) == "[0.5000000,-1.0000000,0.0000000]"


def test_vec_literal_none_passthrough():
    assert retrieval._vec_literal(None) is None
