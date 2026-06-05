# Data flow

The core rule: **one dataset is never copied between folders.** Source images
live in exactly one canonical place; the tile dataset is a hash-addressed cache;
a run stores only results + references.

```
DCS capture (host)
   ▼  writes directly
STORAGE_ROOT/datasets/raw/{dataset_id}/
   ├── images/                         # the ONLY copy of source images
   ├── metadata/
   ├── annotations/{source.coco.json, instances_{train,val,test}.json}
   ├── classes.json
   ├── dataset_manifest.json
   └── capture_manifest.json
   ▼  register (SQLite)              ▼  hash(dataset) + hash(split) + hash(tile_config)
STORAGE_ROOT/cache/tiles/{dataset_hash}__{split_hash}__{tile_config_hash}/
   ├── images/{train,val,test}/
   ├── instances_{train,val,test}.json
   ├── tile_manifest.json
   ├── dropped_tiles_manifest.jsonl
   └── cache_meta.json                 # reused if present & valid (status=reused)
   ▼  train reads data_dir = tile cache
STORAGE_ROOT/runs/{run_id}/
   ├── run_config.yaml, config_snapshot.yaml
   ├── status.json, events.jsonl       # live tracking
   ├── logs/ checkpoints/ metrics/ exports/ reports/
```

Registry (`STORAGE_ROOT/metadata/pipeline.db`): `datasets`, `images`,
`annotations`, `pipeline_runs`, `artifacts`, `model_checkpoints`,
`evaluation_reports`, and (Stage 4) `tile_caches`.

## Forbidden patterns (removed from the legacy project)

- ❌ `storage/shared/dcs_source` as a mandatory copy stage
- ❌ `storage/tmp/{run_id}/enriched_dataset`
- ❌ `cp -r` of a full dataset between stages
- ❌ a permanent duplicate in `DCS_AutoDataset/dataset`

## Implementation status

| Piece | Stage |
|---|---|
| canonical `datasets/raw/{dataset_id}` | ✅ Stage 2 — DCS writes to ephemeral `datasets/_staging/`; the adapter **copies** images into raw by default (hardlink is opt-in with a startup self-check; never a silent fallback). Staging is cleaned via `--cleanup-staging`. No `shared/dcs_source` store. |
| content-addressed `cache/tiles/{dataset_hash}__{split_hash}__{tile_config_hash}` | ✅ Stage 3 — `build-cache` builds/reuses; atomic `.building`→rename; `cache_meta.json` (status/schema/layout/hashes) gates reuse; YOLOX `data_dir` = cache root; breadcrumb `runs/{run_id}/tile_cache.json`. Replaces `tmp/enriched_dataset`. |
| `runs/{run_id}/status.json` + `events.jsonl` | ✅ Stage 4 — `status_tracker` (atomic merge writes, append-only events, UTC, `event_schema_version`); strict for critical ops, best-effort for iter-progress |
| `tile_caches` registry table + `pipeline_runs.{current_stage,model_name,tile_cache_id}` | ✅ Stage 4 — idempotent `init_db` migration (PRAGMA-guarded ALTER); `register_tile_cache` (INSERT OR REPLACE) |
