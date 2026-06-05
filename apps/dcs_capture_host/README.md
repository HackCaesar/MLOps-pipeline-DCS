# DCS AutoDataset

Проект генерирует синтетический датасет из DCS World: запускает миссии, ставит камеру на корабельную высоту `15 m ASL`, делает screenshot, собирает телеметрию объектов, строит bbox, уточняет bbox по видимым пикселям и экспортирует разметку в `YOLO` и `COCO`.

Главная идея: не использовать DCS geometry bbox как финальную разметку. Local bbox из DCS нужен только как начальная геометрическая подсказка. Финальный training bbox берется из `VisibleBBoxRefiner`, который ищет видимые пиксели объекта на реальном screenshot.

## Что Сейчас Умеет
- Запускать DCS с `.miz` миссией.
- Ставить Lua hooks в `Saved Games/DCS/Scripts`.
- Делать token-based sync между telemetry snapshot и screenshot.
- Делать pause-first sync: сначала ставить DCS на паузу, потом разрешать snapshot + screenshot.
- Планировать камеру на высоте `15 m` над уровнем моря.
- Держать камеру горизонтально: `pitch ~= 0`.
- Проецировать DCS local bbox объекта в пиксели через `ObjectLocalBBox`.
- Уточнять bbox до видимого объекта через image-based refinement.
- Отбрасывать сомнительные объекты вместо экспорта плохой разметки.
- Писать:
  - `dataset/images/*.png`
  - `dataset/metadata/*.json`
  - `dataset/debug_overlays/*_overlay.png`
  - `dataset/yolo_labels/*.txt`
  - `dataset/annotations/_annotations.coco.json`

## Важные Директории
- `configs/` - YAML конфиги для обычного запуска и bbox-test запуска.
- `orchestrator/` - основной Python pipeline.
- `dcs_export/` - Lua скрипты для DCS: telemetry, screenshot, camera request.
- `projection/` - математика world-to-camera-to-image.
- `validators/` - visible bbox refinement и пересчет старого датасета.
- `exporters/` - YOLO/COCO export.
- `scenario_builder/` - генерация `.miz` миссий через `pydcs`.
- `missions/` - готовые и сгенерированные миссии.
- `dataset/` - результат генерации.
- `logs/` - логи pipeline.

## Установка На Новом Устройстве
1. Установить DCS World.
2. Установить Python 3.11+.
3. Скопировать проект, например в `D:\Ai_copy\DCS_AutoDataset`.
4. Создать venv:
```powershell
cd D:\Ai_copy\DCS_AutoDataset
python -m venv D:\Ai_copy\.venv
D:\Ai_copy\.venv\Scripts\python.exe -m pip install --upgrade pip
D:\Ai_copy\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
5. Открыть `configs/config.yaml` и проверить пути:
```yaml
paths:
  dcs_exe: "D:/Steam/steamapps/common/DCSWorld/bin/DCS.exe"
  saved_games_dir: "C:/Users/<user>/Saved Games/DCS"
  scripts_dir: "C:/Users/<user>/Saved Games/DCS/Scripts"
  screenshots_dir: "C:/Users/<user>/Saved Games/DCS/ScreenShots"
  output_root: "./dataset"
```
6. Проверить dry-run:
```powershell
D:\Ai_copy\.venv\Scripts\python.exe D:\Ai_copy\DCS_AutoDataset\orchestrator\main.py --config D:\Ai_copy\DCS_AutoDataset\configs\config.yaml --dry-run
```

## Быстрый Preflight
В `D:\Ai\run_pipeline_check.ps1` добавлен wrapper для проверки проекта. Он компилирует Python файлы и запускает dry-run.

Команда:
```powershell
cd D:\Ai
.\run_pipeline_check.ps1 -PreflightOnly
```

Также работает старое написание с опечаткой:
```powershell
.\run_pipeline_check.ps1 -PerflightOnly
```

## Обычный Запуск Одной Миссии
```powershell
D:\Ai_copy\.venv\Scripts\python.exe D:\Ai_copy\DCS_AutoDataset\orchestrator\main.py `
  --config D:\Ai_copy\DCS_AutoDataset\configs\config.yaml `
  --mission D:\Ai_copy\DCS_AutoDataset\missions\mvp_ship_only.miz `
  --frames 3 `
  --frame-id ship_test
```

Результаты будут:
- `dataset/images/ship_test_0001.png`
- `dataset/metadata/ship_test_0001.json`
- `dataset/debug_overlays/ship_test_0001_overlay.png`
- `dataset/yolo_labels/ship_test_0001.txt`
- обновленный `dataset/annotations/_annotations.coco.json`

## BBox Regression Test
Для проверки разных миссий, объектов и погоды используется отдельный конфиг:
- `configs/bbox_test_capture.yaml`

Он специально включает:
```yaml
runtime:
  close_dcs_on_exit: true
  kill_existing_dcs_before_launch: true
  kill_dcs_on_exit: true
```

Это важно: DCS должен закрываться перед каждой новой `.miz`, иначе новый запуск может не заменить миссию и будет продолжаться старая сцена.

Сгенерировать 5 тестовых миссий:
```powershell
D:\Ai_copy\.venv\Scripts\python.exe D:\Ai_copy\DCS_AutoDataset\scenario_builder\create_bbox_test_missions.py `
  --output-dir D:\Ai_copy\DCS_AutoDataset\missions\bbox_tests
```

Запустить одну тестовую миссию:
```powershell
D:\Ai_copy\.venv\Scripts\python.exe D:\Ai_copy\DCS_AutoDataset\orchestrator\main.py `
  --config D:\Ai_copy\DCS_AutoDataset\configs\bbox_test_capture.yaml `
  --mission D:\Ai_copy\DCS_AutoDataset\missions\bbox_tests\bbox_test_01_clear_mixed.miz `
  --frames 3 `
  --frame-id bbox_test_01
```

## Как Идет Один Capture
1. Python pipeline создает `frame_token`.
2. Если включен `probe_before_capture`, сначала делается probe snapshot без screenshot.
3. Pipeline выбирает первичный target и планирует camera pose.
4. Camera pose пишется в `cv_camera_request.json`.
5. DCS Lua export применяет камеру и пишет `cv_camera_ack.json`.
6. Python пишет capture request со статусом `pause_pending`.
7. `Hooks.lua` ставит DCS на паузу и пишет `cv_pause_ack.json`.
8. Только после pause ack Python переводит request в `pending`.
9. `Hooks.lua` обновляет `cv_bbox_cache.txt` из `unit:getDesc().box`.
10. `Hooks.lua` синхронно просит Export environment записать snapshot с `world_basis_raw.axis_x/y/z`, `bbox_min_local`, `bbox_max_local`.
11. `Hooks.lua` делает screenshot, пишет screenshot ack и снимает паузу.
12. Python проверяет pause/token sync.
13. Python строит geometry bbox proposal через `ObjectLocalBBox` + `project_local_bbox`.
14. Если DCS bbox отсутствует, Python использует `model_dims.json` + `ObjectCuboid` как fallback.
15. `VisibleBBoxRefiner` уточняет bbox по видимым пикселям screenshot.
16. Exporter пишет JSON, YOLO, COCO и overlay.

## Почему Есть Geometry BBox И Visible BBox
Geometry bbox строится в первую очередь из local bbox DCS (`desc.box` -> `cv_bbox_cache.txt` -> `bbox_min_local`/`bbox_max_local`). Он полезен как безопасное предложение, но часто слишком большой: включает пустоты вокруг мачт, крыльев, хвоста, корпуса под углом и т.д. `model_dims.json` и `ObjectCuboid` используются только как fallback, если DCS bbox отсутствует.

Visible bbox строится по изображению. Это ближе к тому, что нужно train detector: bbox вокруг видимой части объекта в пикселях.

В metadata сохраняются оба:
- `projection.bbox_xyxy_geometry_px` - исходная 3D-проекция.
- `projection.bbox_xyxy_visible_px` - уточненный bbox.
- `projection.bbox_xyxy_px` - bbox, который идет в YOLO/COCO.

## Что Значит `exact_visible`
`quality_tier: exact_visible` означает, что объект прошел visible-pixel refinement и его bbox можно экспортировать для обучения.

Если refinement не уверен, объект получает:
- `quality_tier: reject`
- `validation.valid: false`
- `projection.bbox_xyxy_px: null`

Такие объекты не попадают в YOLO/COCO. Это сделано намеренно: лучше потерять сомнительный объект, чем обучать модель на неправильной рамке.

## Ограничения
- BBox axis-aligned, поэтому у корабля bbox может включать пустое небо между мачтой и корпусом. Это нормально для прямоугольной bbox-разметки.
- Image-based refinement не является настоящей instance mask из движка. Он сильно лучше грубой 3D projection, но не равен pixel-perfect segmentation.
- Если нужен 100% pixel-perfect bbox, следующий шаг - engine-side render pass с instance mask/depth.
- Некоторые маленькие или плохо видимые самолеты/вертолеты будут rejected.
- Текущая интеграция DCS рассчитана на Windows Python, потому что DCS запускается как Windows process.

## Главные Файлы Для Чтения
- `orchestrator/main.py` - CLI + `PipelineOrchestrator` (тонкий координатор).
- `orchestrator/metadata_builder.py` - схема `dataset/metadata/*.json`, правила validation/quality_tier.
- `orchestrator/dcs_exchange.py` - file protocol Python ↔ Lua.
- `orchestrator/camera_planner.py` - выбор target + planning pose на 15 m ASL.
- `validators/visible_bbox_refiner.py` - уточнение bbox по screenshot.
- `projection/projector.py` - проекция 3D в 2D.
- `exporters/coco_yolo.py` - запись YOLO/COCO.
- `scenario_builder/create_bbox_test_missions.py` - 5 regression-test миссий.
- `dcs_export/Export.lua` - telemetry и camera control.
- `dcs_export/Hooks.lua` - screenshot hook.

Подробности — `project_structure.md` и `README.md` каждого подкаталога.

## Regression Harness (без DCS)

После любого изменения в core pipeline запускать:

```powershell
python -m pytest tests/unit/ -q          # 21 unit-тест projector-а
python scripts/run_regression.py         # 6-кадровая регрессия refine + export
```

Регрессия НЕ запускает DCS — она работает на сохранённых PNG + metadata из `tests/regression/fixtures/`. Подробности — `tests/regression/README.md`.
