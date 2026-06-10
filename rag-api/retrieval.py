"""Semantic search over calls/utterances (read path).

Ported from call-analysis-api/pipeline/persistence.py — the write path no longer
serves search, so these live here in rag-api.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ragcore import db, embeddings


def _vec_literal(vec: Optional[List[float]]) -> Optional[str]:
    if vec is None:
        return None
    return "[" + ",".join(f"{v:.7f}" for v in vec) + "]"


def search_calls(*, query: str, top_k: int = 5,
                 min_similarity: float = 0.0) -> List[Dict[str, Any]]:
    """Top-k calls by cosine similarity of summary_embedding to ``query``."""
    if not query.strip():
        return []
    qvec_literal = _vec_literal(embeddings.embed_one(query))
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    call_id, file_name, received_at, duration_sec,
                    closer_name, closer_confidence, client_name,
                    closer_talk_ratio, closer_wpm, client_wpm,
                    objection_count, buying_signal_count, next_step_secured,
                    rapport_quality, predicted_outcome,
                    summary, pdf_url, html_url,
                    1 - (summary_embedding <=> %s::halfvec) AS similarity
                FROM calls
                WHERE summary_embedding IS NOT NULL
                  AND 1 - (summary_embedding <=> %s::halfvec) >= %s
                ORDER BY summary_embedding <=> %s::halfvec
                LIMIT %s
            """, (qvec_literal, qvec_literal, min_similarity, qvec_literal, top_k))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def search_utterances(*, query: str, top_k: int = 10,
                      min_similarity: float = 0.0,
                      role: Optional[str] = None) -> List[Dict[str, Any]]:
    """Top-k utterances by cosine similarity to ``query``. Optional role filter."""
    if not query.strip():
        return []
    qvec_literal = _vec_literal(embeddings.embed_one(query))
    sql = """
        SELECT
            call_id, idx, start_sec, end_sec, speaker, role,
            text, sentiment_label, emotion_label, valence, arousal, dominance,
            1 - (text_embedding <=> %s::halfvec) AS similarity
        FROM utterances
        WHERE text_embedding IS NOT NULL
          AND 1 - (text_embedding <=> %s::halfvec) >= %s
    """
    params: list = [qvec_literal, qvec_literal, min_similarity]
    if role:
        sql += " AND role = %s "
        params.append(role)
    sql += " ORDER BY text_embedding <=> %s::halfvec LIMIT %s "
    params.extend([qvec_literal, top_k])
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
