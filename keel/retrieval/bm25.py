"""Lexical retrieval over the `chunks_fts` FTS5 table, ranked by SQLite's bm25().

`sanitise_query()` turns arbitrary user text into a safe FTS5 expression: word and number tokens are
double-quoted and joined with OR, so operators, stray quotes and parentheses in the input can never
raise a syntax error. ACL filtering and the quarantine exclusion happen inside the SQL query: the
user's tags are bound as one JSON array (any number of tags, one host parameter), and a chunk counts
only when its `acl_tags` column holds a JSON array sharing a value with them; a row whose column is
malformed or is not an array is visible to nobody rather than an error for everybody.
"""

from __future__ import annotations

import json
import re
import sqlite3

from keel.providers.base import ChunkHit
from keel.providers.local_index import HIT_COLUMNS, hit_from_row

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
MAX_QUERY_TOKENS = 64

# The chunk's tag list as json_each can read it: an empty array when the column is malformed JSON or
# any JSON shape other than an array, so such rows match no user.
_CHUNK_TAGS = (
    "CASE WHEN json_valid(c.acl_tags) THEN "
    "(CASE WHEN json_type(c.acl_tags) = 'array' THEN c.acl_tags ELSE '[]' END) ELSE '[]' END"
)
_ACL_CLAUSE = (
    f" AND EXISTS (SELECT 1 FROM json_each({_CHUNK_TAGS}) AS chunk_tag"
    " WHERE chunk_tag.type = 'text' AND chunk_tag.value IN (SELECT value FROM json_each(?)))"
)


def _has_control_char(tag: str) -> bool:
    return any(ord(ch) < 0x20 or ch == "\x7f" for ch in tag)


def sanitise_query(query: str, *, max_tokens: int = MAX_QUERY_TOKENS) -> str:
    """Return an FTS5 MATCH expression of quoted tokens joined by OR, or an empty string when the query
    holds no word characters."""
    tokens = _TOKEN.findall(query or "")[:max_tokens]
    return " OR ".join(f'"{token}"' for token in tokens)


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    k: int,
    allowed_tags: list[str] | None = None,
) -> list[ChunkHit]:
    """Top-k non-quarantined chunks by BM25 for `query`. `score` is the negated bm25() rank, so higher is
    better. With `allowed_tags`, only chunks sharing at least one tag are returned; an empty list matches
    nothing; None applies no ACL filter."""
    if k <= 0:
        return []
    match = sanitise_query(query)
    if not match:
        return []
    params: list[object] = [match]
    acl_clause = ""
    if allowed_tags is not None:
        # A tag holding a control character can never name a stored tag, and SQLite's JSON reader
        # would cut a NUL-bearing value short ("hr\x00" reads as "hr"), so such tags are dropped.
        tags = sorted({str(t) for t in allowed_tags if not _has_control_char(str(t))})
        if not tags:
            return []
        acl_clause = _ACL_CLAUSE
        params.append(json.dumps(tags))
    params.append(int(k))
    sql = (
        f"SELECT {HIT_COLUMNS}, bm25(chunks_fts) AS bm25_rank "
        "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid JOIN documents d ON d.id = c.document_id "
        f"WHERE chunks_fts MATCH ? AND c.quarantined = 0{acl_clause} "
        "ORDER BY bm25_rank, c.id LIMIT ?"
    )
    rows = conn.execute(sql, params).fetchall()
    return [hit_from_row(row, -float(row["bm25_rank"])) for row in rows]
