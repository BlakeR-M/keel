# Adversarial security review

Before Keel was first published, three independent reviewers were pointed at the security-bearing
paths with one instruction: break them. This page is the result, with the severity of each finding
and the test that now proves the fix.

Every attack is a test in the three red-team files, which `pytest` collects by default:

- [`tests/redteam_acl_and_retrieval_leakage.py`](../tests/redteam_acl_and_retrieval_leakage.py)
- [`tests/redteam_tool_policy_and_approvals.py`](../tests/redteam_tool_policy_and_approvals.py)
- [`tests/redteam_ledger_integrity_injection_screening_and_air_gap.py`](../tests/redteam_ledger_integrity_injection_screening_and_air_gap.py)

Twenty-seven findings came out of the review. Twenty-five are fixed, each row below naming the test
that proves it. Two stay open as strict `xfail(strict=True)` by choice, with the reason written into
the test and the control that covers it: one medium (a paraphrased injection carrying no trigger
word the heuristics key on, covered by the LLM judge at ingest, `keel ingest --judge`) and one low
(a heuristic false positive an operator releases from the admin quarantine list).

| # | Finding | Severity | Status | Proof |
|---|---------|----------|--------|-------|
| L1 | Ledger.append called a Python UDF inside `sqlite3_step`; a reader on the shared connection froze the whole process | critical | fixed: hash computed in Python under `BEGIN IMMEDIATE` and a per-connection lock (`keel/safety/ledger.py`) | `redteam_ledger...::test_ledger_append_alongside_a_reader_on_the_shared_connection_completes` |
| A1 | Beyond loopback, self-asserted `tags` in the body or query string widened access; no trusted identity channel | high | fixed: `trusted_identity` in `keel/web/app.py`; beyond loopback identity comes only from `X-Keel-User`/`X-Keel-Tags` with `X-Keel-Proxy-Token` = `KEEL_PROXY_TOKEN`, everyone else runs as `public` | `redteam_acl...::test_a11_...`, `::test_a12_...` |
| P1 | Calculator stacked powers built unbounded integers (`((2**4096)**4096)**4096` never returned) | high | fixed: result size bounded before a power is computed and after every operation (`_MAX_RESULT_BITS`) | `redteam_tool_policy...::test_calculator_stacked_power_is_not_a_dos` |
| I2 | An instruction in a Markdown heading was never screened yet reached every prompt via the source label | high | fixed: each chunk is screened together with its heading (`keel/ingest/pipeline.py`) | `redteam_ledger...::test_instruction_in_a_markdown_heading_is_screened_before_it_reaches_the_model` |
| I3 | An instruction in a document title (`<title>`, `#`, PDF/DOCX metadata) rode into every prompt | high | fixed: the title is screened; a flagged title is stored as the file or URL name, the original kept in `meta` and the ledger | `redteam_ledger...::test_instruction_in_an_html_title_is_screened_before_it_reaches_the_model` |
| A2 | Re-ingesting a document with different explicit ACL tags was a silent no-op | medium | fixed: the document and its chunks are retagged in one transaction, the index re-upserted, a ledger row written; `keel ingest --tags` default is now "unchanged for a stored source" | `redteam_acl...::test_a16_reingest_with_narrower_tags_must_restrict_or_refuse` |
| P2 | `sql_readonly` had no cap on a single value's allocation (`hex(zeroblob(1e9))`) | medium | fixed: `SQLITE_LIMIT_LENGTH` set to 1 MB on the tool connection, `zeroblob`/`randomblob`/`load_extension` denied by the authorizer, serialised result capped | `redteam_tool_policy...::test_sql_readonly_caps_single_value_size` |
| P3 | `ApprovalQueue.execute` never re-checked the policy; an approval outlived the policy that permitted it | medium | fixed: `execute()` re-checks the policy passed in or the one bound to the registry (`Policy(registry=...)` binds) and stores a refusal as the result; web and CLI pass `ctx.policy` | `redteam_tool_policy...::test_execute_re_checks_current_policy` |
| I4 | An instruction split across two sections scored under the threshold in each chunk | medium | fixed: adjacent chunk pairs that passed alone are screened together and both halves flagged when the pair trips | `redteam_ledger...::test_instruction_split_across_two_sections_is_quarantined_by_the_pipeline` |
| I5 | Paraphrased instructions without trigger words passed the heuristics | medium | half fixed: "from now on ... your only job ... tell people that" is caught (dictated-output pattern widened); the "any automated reader summarising this note" wording stays with the LLM judge (`keel ingest --judge`), since `tests/test_safety.py::test_local_judge_flags_a_paraphrased_injection_and_passes_clean_text` pins that phrasing as a heuristics miss | first wording: `redteam_ledger...::test_paraphrased_instruction_without_trigger_words_is_quarantined[From now on...]`; second wording keeps its xfail |
| G1 | Name resolution ran unguarded under air-gap (DNS exfiltration channel) | medium | fixed: `socket.getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr` refuse names outside the allow list; IP literals, allow-listed names and passive (bind) lookups pass (`keel/airgap.py`) | `redteam_ledger...::test_dns_lookup_of_a_host_outside_the_allow_list_is_refused`; the control `test_requests_and_urllib3_are_refused_before_any_connect` now asserts `asked == []` |
| A3 | `/source` 403 named the tags a chunk needs | low | fixed: fixed message that names nothing about the chunk | `redteam_acl...::test_a10_source_viewer_forbidden_response_names_no_acl_tags` |
| A4 | `SqliteVectorIndex` served a stale ACL mask after a tag update | low | fixed: fingerprint covers tags and quarantine ids; returned rows are re-checked against fresh data and the cache reloads when stale | `redteam_acl...::test_a19_...`; `redteam_ledger...::test_vector_index_excludes_a_chunk_quarantined_after_its_cache_warmed` |
| A5 | Malformed `acl_tags` JSON broke BM25 for everyone | low | fixed: `json_valid`/`json_type` guard, such rows are visible to nobody | `redteam_acl...::test_a05_...` |
| A6 | Oversized tag list crashed BM25 (`too many SQL variables`) | low | fixed: tags bound as one JSON array | `redteam_acl...::test_a20_...` |
| A7 | No ACL re-check at the generation boundary | low | fixed: `AnswerEngine.answer` and `search_docs` re-check `allowed(hit, user.tags)` | `redteam_acl...::test_a21_...` |
| P4 | Calculator returned complex numbers | low | fixed: non-real results refused | `redteam_tool_policy...::test_calculator_does_not_return_complex` |
| P5 | Calculator emitted inf/nan | low | fixed: non-finite constants and results refused with ToolError (the test accepts a refusal as the safe outcome) | `redteam_tool_policy...::test_calculator_stays_finite` |
| L2 | Ledger accepted NaN/Infinity, breaking strict-JSON verifiers | low | fixed: `canonical_json(allow_nan=False)` | `redteam_ledger...::test_ledger_refuses_nan_and_infinity_so_the_export_stays_strict_json` |
| L3 | Hash material ambiguous across the kind/request_id boundary | low | fixed before first publish (`ed2012b`): hash material is a canonical JSON array of the four chained fields, so field boundaries stay explicit; docs/architecture.md describes the recipe | `redteam_ledger...::test_ledger_moving_characters_across_the_kind_and_request_id_boundary_is_detected` |
| L4 | `verify_file` choked on a UTF-8 BOM | low | fixed: `utf-8-sig`, decode errors reported as unreadable | `redteam_ledger...::test_ledger_export_resaved_with_a_utf8_bom_still_verifies` |
| I6 | Short base64-encoded instructions passed | low | fixed: runs of 40+ characters are decoded and the plaintext screened | `redteam_ledger...::test_short_base64_encoded_instruction_is_quarantined` |
| I7 | Benign technical prose quarantined by one pattern | low | two of three fixed: setting guidelines/rules/documents aside is a weaker `override_documents` signal, and bare `agent`/`model` count as an AI address only behind an article or a colon; the LLM-ops note that says "reveal the current system prompt" stays a heuristic false positive (release from the admin quarantine list) | `redteam_ledger...::test_benign_technical_prose_stays_available[superseded-guidelines]`, `[keel-docs-web-route-table]`; `[llm-ops-note]` keeps its xfail |
| I8 | Ingest ledger row said nothing about quarantined chunks | low | fixed: the ingest row carries `quarantined`, `quarantined_chunk_ids`, `quarantine_reasons` and `title_replaced` when relevant | `redteam_ledger...::test_ingest_ledger_records_that_chunks_were_quarantined` |
| I9 | Ingest committed before its ledger row and swallowed ledger failures | low | fixed: ledger row written inside the ingest transaction; a failure rolls the ingest back | `redteam_ledger...::test_ingest_is_rolled_back_when_its_ledger_row_cannot_be_written` |
| S1 | ABN/TFN/Medicare/card/phone patterns missed numbers wrapped across a line or tab | low | fixed: one whitespace character or a hyphen between digit groups | `redteam_ledger...::test_abn_and_phone_wrapped_across_a_line_break_are_redacted` |
| S2 | Email pattern was ASCII-only | low | fixed: Unicode word classes | `redteam_ledger...::test_email_with_a_unicode_domain_or_local_part_is_redacted` |

Follow-ups from the pass, all done: docs/web.md carries the proxy identity paragraph
(`KEEL_PROXY_TOKEN`, `X-Keel-User`, `X-Keel-Tags`); docs/architecture.md describes the canonical
JSON array ledger append; `pyproject.toml` collects `redteam_*.py`; L3 was decided and fixed
(`ed2012b`).

