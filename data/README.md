# Enterprise AI Copilot — Starter Dataset
All AcmeOps documents, incident reports, releases, functional specifications, configurations and schema files in this ZIP are **synthetic** and deliberately fictional. No real customers or private data.
## Content
- `documents/`: 12 Markdown user/policy documents for RAG.
- `releases/`: 6 release notes for the release chatbot.
- `incidents/incidents.csv`: 24 synthetic incidents, grouped into 12 labeled duplicate pairs.
- `database/schema_v1.sql`, `schema_v2.sql`: two SQL schema versions to compare.
- `requirements/`: 8 FSDs, an XML configuration, and 4 seed test cases.
## Getting started
1. Ingest documents and releases into Qdrant. Preserve file path and version as metadata.
2. Use `incidents.csv` for semantic search and duplicate detection. Split evaluation by `duplicate_group` to avoid leakage. This dataset is too small to make generalization claims.
3. Compare schema_v1 and schema_v2, then explain changes with an LLM.
4. Use the FSDs and config.xml to detect deliberate contradictions and generate tests.
5. Add real public data later for realistic evaluation.
**Important:** Do not present this synthetic sample as real production data or a robust benchmark.
