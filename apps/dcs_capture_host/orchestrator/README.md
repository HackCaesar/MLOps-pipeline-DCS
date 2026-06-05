# orchestrator

Пакет, который связывает DCS, Lua-hooks, проекцию, refinement и export в один pipeline.

После рефакторинга `main.py` похудел с 1490 до 399 строк за счёт выноса логики в фокусные модули. `PipelineOrchestrator` теперь — тонкий координатор, который композирует компоненты.

## Модули

| Модуль | Ответственность |
|---|---|
| `main.py` | CLI (`--config`, `--mission`, `--frames`, `--dry-run`, etc.) + `PipelineOrchestrator` (один class, который запускает DCS и идёт по кадрам). |
| `runtime_paths.py` | `PROJECT_ROOT`, `resolve_config_path`, `RuntimePaths` dataclass. Единственное место, где config-строки превращаются в `Path`. |
| `json_io.py` | `atomic_write_json` (retry на `PermissionError`), `load_json`, `load_yaml_config`, `wait_for_stable_file`, `utc_now_iso`. Все retry-семантики Windows-friendly file replace. |
| `dcs_exchange.py` | `DCSExchange` — file-protocol клиент. Знает про `cv_capture_request.json`, `cv_camera_request.json`, `cv_*_ack.json`, `snapshot_<token>.json`, screenshot pickup. Не запускает DCS. |
| `camera_context.py` | Базовая математика basis + `build_camera_context(snapshot_camera, camera_config) -> (CameraPose, CameraIntrinsics, camera_state)`. |
| `camera_planner.py` | `CameraPlanner` — выбор primary target и planning camera pose на 15 m ASL с pitch ≈ 0. |
| `metadata_builder.py` | `MetadataBuilder` — собирает frame/object metadata. Владеет всеми именами полей (`projection.bbox_xyxy_px`, `quality_tier`, `validation.*` и т.д.). |
| `overlay.py` | `render_overlay(image_path, frame_metadata, overlay_path)` — debug PNG поверх screenshot. |

## Как запускается

Dry-run (валидация config + создание директорий):
```powershell
python orchestrator/main.py --dry-run
```

Capture одной миссии:
```powershell
python orchestrator/main.py `
  --config configs/config.yaml `
  --mission missions/mvp_ship_only.miz `
  --frames 3 `
  --frame-id test_run
```

CLI-флаги: `--config`, `--mission`, `--frames`, `--frame-id`, `--dry-run`, `--attach`, `--install-only`, `--skip-install`, `--skip-camera-move`.

## Поток одного кадра (`capture_one_frame`)

1. Генерируется `frame_token = uuid4()[:8]`.
2. `clear_artifacts()` — удаляются все exchange + output файлы для этого frame_id/token.
3. (Опционально) `_probe_scene_and_move_camera()`:
   - probe snapshot без screenshot;
   - `CameraPlanner.plan_camera_pose(snapshot)` выбирает позицию и направление;
   - `DCSExchange.request_camera_pose(...)` + `wait_for_camera_ack(...)`;
   - небольшая `camera_settle_delay_s` пауза.
4. `DCSExchange.clear_capture_exchange()` + `request_capture()` — двухстадийный pause-first.
5. `DCSExchange.wait_for_snapshot()` + `wait_for_screenshot()`.
6. `MetadataBuilder.build_frame_metadata(...)` собирает rich metadata.
7. `VisibleBBoxRefiner.refine_frame(image, metadata)` обновляет bbox.
8. `atomic_write_json` пишет `metadata/<frame_id>.json`.
9. `DatasetExporter.add_frame` / `write_yolo` / `write_coco`.
10. `render_overlay(...)` пишет debug PNG.

## Почему один большой `_probe_scene_and_move_camera`

Этот метод — это **orchestration** (clear_artifacts → request → wait → plan → camera_request → wait → settle). Он умышленно держится в `PipelineOrchestrator`, потому что разбивать его на абстракции преждевременно: каждый шаг привязан к одному конкретному порядку с DCS.

## Tokens и pause-first

Подробности: `dcs_export/README.md`. Кратко: каждый запрос несёт `frame_token` или `request_token`. Acks принимаются только при совпадении токена. Pause-first capture двигает request через `pause_pending → pending`.

## Что нельзя менять без Algorithm Change Proposal

- `MetadataBuilder.build_frame_metadata` шаблон — это **публичная схема** `dataset/metadata/*.json`.
- `MetadataBuilder.validate_projection` rules — определяют `validation.valid` и `quality_tier`.
- `DCSExchange.request_capture` две стадии `pause_pending → pending` — Lua hook ожидает именно этот порядок.
- Имена полей в `cv_*_ack.json` и `snapshot_<token>.json` — DCS file protocol.

## Regression и unit-тесты

```powershell
python -m pytest tests/unit/ -q          # покрывает projector
python scripts/run_regression.py         # full pipeline без DCS, 6 фреймов
```

Регрессия НЕ запускает DCS, она реплеит refinement + export на сохранённых PNG/metadata. См. `tests/regression/README.md`.
