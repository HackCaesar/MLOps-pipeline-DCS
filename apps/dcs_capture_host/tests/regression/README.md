# Regression Harness

Этот harness нужен, чтобы рефакторинг не сломал результат pipeline. DCS он не запускает — работает только на сохраненных PNG и metadata.

## Зачем это нужно

В проекте нет unit-тестов и нет фиксированной системы regression проверок. Любое изменение в `orchestrator/main.py`, `validators/visible_bbox_refiner.py`, `projection/projector.py` или `exporters/coco_yolo.py` могло тихо изменить bbox, YOLO labels или COCO annotations.

Этот harness фиксирует текущий output на маленьком наборе кадров как golden и после каждого refactor-step сравнивает новый output с golden.

## Что покрывается

6 кадров из `dataset/` подобраны так, чтобы покрыть разные кейсы:

| Frame | Назначение |
|---|---|
| `regression_01_f01` | стандартная сцена, 4 valid объекта (ships + airplanes + helicopters) |
| `regression_01_f02` | другой кадр той же сцены |
| `regression_03_f01` | 5 объектов, 1 rejected — покрывает reject branch |
| `regression_05_f01` | другая миссия (sunset), 5 valid |
| `regression_05_f03` | другой ракурс той же миссии |
| `pose_sync_unique_03` | одиночный sync sanity check |

В сумме: **6 frames, 6 COCO images, 26 COCO annotations**.

Структура:

```
tests/regression/
├── fixtures/
│   ├── images/<frame_id>.png            # input PNG для refiner
│   └── metadata_input/<frame_id>.json   # input metadata (после refinement; refiner идемпотентен)
└── golden/
    ├── metadata/<frame_id>.json         # эталон после refinement
    ├── yolo_labels/<frame_id>.txt       # эталон YOLO
    └── annotations/_annotations.coco.json  # COCO только по 6 fixture-кадрам
```

## Как пользоваться

### Обычная проверка (check mode)

```powershell
python scripts/run_regression.py
```

Запускает `VisibleBBoxRefiner.refine_frame` + `DatasetExporter` на 6 fixtures, сравнивает с golden. Возвращает exit code `0` при совпадении, `1` при diff. Все различия печатаются в stdout.

### Обновление golden

```powershell
python scripts/run_regression.py --update-golden
```

Перезаписывает golden output из текущего кода. **Использовать только после осознанного изменения** (Algorithm Change Proposal, новый класс, изменение config-схемы и т.п.). После обновления — сразу `git diff tests/regression/golden/` для ревью изменений.

### Inspect-режим (сохранить generated output)

```powershell
python scripts/run_regression.py --keep-output ./_regression_out
```

Кладёт metadata/YOLO/COCO в указанный каталог рядом с debug-PNG для ручного просмотра.

## Что именно проверяется

- **metadata JSON** — deep dict diff с float tolerance `abs=1e-6`, `rel=1e-9`. Включая: `bbox_xyxy_px`, `bbox_xyxy_unclipped_px`, `bbox_xyxy_geometry_px`, `bbox_xyxy_visible_px`, `bbox_xywh_px`, `bbox_area_px2`, `center_px`, `quality_tier`, `confidence_source`, `validation.*`, `object_stats`, `usable`, `invalid_reasons`, `sync.*`, `camera.*`, `projection.*`, `visible_bbox_refinement.*`.
- **YOLO labels** — построчное сравнение: первый токен (`class_id`) строго равен, остальные — float c tolerance. Format-only различия (например `0.500000` vs `0.5`) выводятся отдельно как warning.
- **COCO** — deep diff `images`, `annotations`, `categories` (включая порядок и `id`).

## Что игнорируется намеренно

Абсолютные пути к debug-PNG в `visible_bbox_refinement.*_debug_artifacts.*_png` нормализуются: tmpdir-префикс отбрасывается, сравнивается только хвост `debug_refiner/...`. Это потому что:

- абсолютный путь зависит от tmpdir каждого запуска (`%TEMP%/dcs_regression_<rand>/...`);
- хвост пути всё ещё содержит имя выбранного refinement-метода (`__ship_edge_conservative_refinement__overlay.png`), поэтому смена логики refiner-а всё равно будет видна как diff.

## Если регрессия зафейлилась

1. **Не скрывай отличие.** Не запускай `--update-golden`, не разобравшись в diff.
2. Прочитай diff целиком. Каждая строка имеет вид `path: current != golden`.
3. Классифицируй diff:
   - **format-only** (`10` vs `10.0`, разный порядок ключей) — допустимо при refactor, можно обновить golden;
   - **algorithmic** (изменился bbox, quality_tier, validation, class_id) — это потенциально баг или Algorithm Change. Не обновляй golden, не разобравшись;
   - **schema** (новые/исчезнувшие ключи) — refactor должен был сохранить публичный JSON. Восстанови либо вынеси в Algorithm Change Proposal.
4. Только после явного решения по diff — обновляй golden через `--update-golden`.

## Команды после каждого refactor-step

```powershell
python -m compileall .
python -c "from projection import CameraIntrinsics, CameraPose, project_local_bbox, project_cuboid, ObjectLocalBBox, ObjectCuboid"
python -c "from exporters.coco_yolo import DatasetExporter"
python -c "from validators import VisibleBBoxRefiner"
python -c "import orchestrator.main"
python orchestrator/main.py --dry-run --config configs/config.yaml   # на DCS-машине
python scripts/run_regression.py
```

`--dry-run` работает только там, где пути из config валидны (DCS-машина). Сам regression-harness DCS не требует.

## Ограничения

- Harness покрывает **post-snapshot** часть pipeline: refinement + export. DCS-stage (`launch_dcs`, Lua sync, camera planning, projection из snapshot) этим harness-ом не проверяется. Для них нужен ручной smoke на DCS-машине.
- Если меняется input snapshot schema (`world_basis_raw`, `bbox_min_local`, и т.п.) — input metadata в fixtures тоже надо обновить, потому что текущий input — это already-refined metadata.
- Если меняется `generation.allowed_classes` порядок — `class_id` в YOLO/COCO сменится, regression зафейлится. Это правильно: такое изменение требует Algorithm Change Proposal.
