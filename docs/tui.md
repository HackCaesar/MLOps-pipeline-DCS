# Terminal monitoring (TUI)

Monitoring is terminal-based (Rich) — no web UI. The TUI reads
`STORAGE_ROOT/runs/{run_id}/status.json` and `events.jsonl`; it does not talk to
the training process directly, so it works for local and dockerized runs alike.

Implemented in `apps/pipeline/tui.py` + `apps.pipeline.cli` (`run --watch`, `tui`,
`status`). `rich` is optional (`pip install '.[tui]'`): the live view needs it, but
`status` always works in plain text. The viewer reads files only — it never touches
the training process or SQLite. Use `--ascii` on consoles without UTF-8.

## Commands

```bash
# run the pipeline and render live status until it finishes
python -m apps.pipeline.cli run \
  --config configs/pipeline_production.yaml \
  --dataset-id dcs_caucasus_100 --run-id prod_2026_05_31_001 --watch

# attach a live TUI to a run (default: latest run under storage.runs_dir)
python -m apps.pipeline.cli tui    --config configs/pipeline_production.yaml --run-id prod_2026_05_31_001

# one-shot snapshot (works without rich; --json for raw status.json)
python -m apps.pipeline.cli status --config configs/pipeline_production.yaml --run-id prod_2026_05_31_001
```

## Example

```
DCS → YOLOX Pipeline

Run:        prod_2026_05_31_001
Dataset:    dcs_caucasus_100
Tile cache: reused
Model:      YOLOX-S      Classes: 3      Device: CUDA

[✓] validate_raw_dataset
[✓] register_dataset
[✓] build_or_reuse_tile_cache
[▶] train              Epoch 4/10 | Iter 54/112 | loss=3.21 | ETA 00:07:20
[ ] evaluate
[ ] export_onnx
[ ] register_artifacts
[ ] finalize

Last events:
- train: Epoch 4/10
- train: checkpoint saved
```
