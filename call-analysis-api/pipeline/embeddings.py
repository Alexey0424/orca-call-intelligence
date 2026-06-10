"""Re-export of the shared embeddings client (single source of truth in ragcore).

The write path (persistence) and the read path (rag-api) must use the identical
MRL-truncation recipe, so it lives in ``ragcore.embeddings``.
"""
from ragcore.embeddings import (  # noqa: F401
    embed_one, embed_one_safe, embed_batch_safe,
)
