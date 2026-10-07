# 4.2i-b — _KNOWN_GAPS fix (DEFERRED)

## Findings (verified)
- `_KNOWN_GAPS` at tools/audit.py:14-15 — 5 items
- Docstring at tools/audit.py:157-158 — 6 items (stale: parallel_tool_calls)
- Read sites: L140 (into _collect), L215 (display as "not implemented")

## Truth table
| Capability | Local | Cloud |
|-----------|-------|-------|
| vision    | yes   | NO (deliberate — routes/chat.py:450-452) |
| audio     | NO    | NO    |
| video     | NO    | NO    |

## Recommended fix (Option 1 — evidence-based)
_KNOWN_GAPS = (
    "vector_memory",
    "cost_tracking_cumulative",  # rate exists, cumulative missing
    "eval_harness",
    "self_modification",
    "cloud_vision_input",        # local yes, cloud no
    "audio_input",               # local+cloud no
)
+ fix docstring L157-158

## Test coverage
- tools/audit.py: 0 tests
- vision tests exist: tests/test_routes.py, test_streaming.py, test_agent_stream_cloud.py

## Not yet done
- Check if any test asserts "gaps:" string
- Decide Option 1 vs 2 vs 3
- PyEOF not written
