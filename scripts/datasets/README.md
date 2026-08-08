# Preloaded datasets

The catalog in `data/preloaded_datasets.json` contains public dataset sources and their provenance. Use the importer from the repository root:

```powershell
python scripts/datasets/preload.py tinystories --limit 1000
python scripts/datasets/preload.py smoltalk --limit 1000
```

The importer downloads rows through the public Hugging Face dataset viewer API, writes normalized JSONL under `storage/datasets/`, and registers the dataset in the IVX database with `status=ready`. Each record includes the source URL, license, retrieval timestamp, requested size, and SHA-256 checksum in `lineage_json`.

Use `--overwrite` to refresh an existing catalog entry. `--limit` is bounded to 100,000 rows and defaults to 1,000. The `c4-en` entry is intentionally opt-in: the official card lists the English corpus at roughly 305 GB and notes that Common Crawl terms and possible PII also apply.

Licenses must be reviewed before redistribution or commercial training. SmolTalk's newly generated subsets are Apache-2.0, while included public subsets retain their original licenses. TinyStories is listed as CDLA-Sharing-1.0.
