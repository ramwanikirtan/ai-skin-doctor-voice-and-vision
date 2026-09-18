# Dermatology Knowledge Base

General dermatology reference passages for the AI Skin Doctor's RAG pipeline.
**Not patient data** — never store voice, images, videos, or personal information here.

## Source allowlist (reputable only)

- American Academy of Dermatology (AAD) — https://www.aad.org
- NHS (UK National Health Service) — https://www.nhs.uk
- Mayo Clinic — https://www.mayoclinic.org
- MedlinePlus / U.S. National Library of Medicine — https://medlineplus.gov
- CDC — https://www.cdc.gov (where relevant)
- WHO — https://www.who.int (where relevant)

Do NOT add blogs, Reddit, commercial skincare sites, or other unreliable sources.

## How to add a document

1. Summarize (in your own words) a page from an allowlisted source — do not paste large verbatim copies.
2. Save as `documents/<topic>.md` with this header:

```
---
title: <page/topic title>
source_org: <e.g. American Academy of Dermatology>
source_url: <canonical page URL>
date_added: <YYYY-MM-DD>
---
```

3. Use `##` section headings — ingestion preserves them as chunk `section` metadata.
4. Record the row in the table below.

## Documents

| File | Title | Organization | Source URL | Added | Status |
|---|---|---|---|---|---|
| `documents/eczema_overview.md` | Eczema (atopic dermatitis) overview | American Academy of Dermatology | https://www.aad.org/public/diseases/eczema | 2026-09-18 | seed |
| `documents/contact_dermatitis_overview.md` | Contact dermatitis overview | NHS / American Academy of Dermatology | https://www.nhs.uk/conditions/contact-dermatitis/ | 2026-09-18 | seed |
| `documents/when_to_see_dermatologist.md` | Warning signs — when to seek care | Mayo Clinic | https://www.mayoclinic.org/diseases-conditions/dermatitis-eczema/symptoms-causes/syc-20352380 | 2026-09-18 | seed |

## Pipeline

- `documents/` → `rag/ingest.py` (Step 2, not built yet) → `processed/chunks.json`
- `processed/` holds generated files only — do not edit by hand.
