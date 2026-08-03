# Machine-Readable Catalog

This directory contains the structured catalog behind the curated method list.

- [`papers.json`](papers.json) is the source of truth for machine-readable metadata.
- [`papers.csv`](papers.csv) is generated for spreadsheets and data analysis.
- The main [`README.md`](../README.md) remains the editorial view, with longer explanations and grouping.

The catalog currently contains 75 entries: 47 methods from the survey snapshot and 28 later additions or related resources.

## Schema

| Field | Meaning |
|---|---|
| `id` | Stable arXiv or OpenReview identifier. |
| `collection` | Survey snapshot or later repository addition. |
| `update_batch` | Repository update in which the entry was added. |
| `scope` | Editorial scope such as Core, Near-Core, Related Signal, or Evaluation. |
| `setting` | Reasoning RL, Agentic RL, Multi-Agent RL, or an explicitly related setting. |
| `category` | README taxonomy section or update group. |
| `granularity` | Token, segment, step, turn, action, agent, or a combination. |
| `methodology` | Main mechanism used to estimate or redistribute credit. |
| `code_url` | Official implementation when verified; empty otherwise. |
| `code_status` | Verification state for code availability. |
| `venue` | Venue stated in the curated README. |
| `publication_status` | Cataloged source status such as arXiv record, OpenReview record, or venue listed; not a quality score. |

The catalog intentionally avoids a single subjective "evidence score." Publication status, code availability, and evaluation scope should be tracked separately as verifiable facts.

## Update And Validate

Edit `papers.json`, update the corresponding curated README entry, and then run:

```bash
python3 scripts/build_catalog.py
python3 scripts/build_catalog.py --check
```

The validator rejects missing required fields, duplicate identifiers or URLs, unsupported paper URLs, stale CSV output, and mismatches between the catalog and the method sections in the README.
