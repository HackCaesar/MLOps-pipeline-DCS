# dcs_export

Каталог содержит Lua интеграцию с DCS World.

## Зачем Нужны Lua Скрипты
Python не может напрямую получить состояние объектов и сделать screenshot из DCS. Поэтому pipeline ставит Lua scripts в `Saved Games/DCS/Scripts`, а Python общается с ними через JSON files в `Saved Games/DCS/Logs`.

## Export.lua
Работает через DCS Export API.

Главные задачи:
- читать `cv_capture_request.json`;
- читать `cv_camera_request.json`;
- применять camera pose через `LoSetCameraPosition`;
- собирать camera state через `LoGetCameraPosition`;
- собирать world objects через `LoGetWorldObjects`;
- читать `cv_bbox_cache.txt` и добавлять local bbox в snapshot;
- предоставлять `CV_DATASET_CAPTURE_NOW(...)` для pause-first snapshot из Hook;
- писать snapshot `cv_snapshots/snapshot_<token>.json`;
- писать `cv_camera_ack.json`.

### Почему Camera Control Здесь Experimental
`LoSetCameraPosition` не является таким же надежным и документированным путем, как обычный telemetry API. Поэтому в коде он помечен как experimental MVP automation. Pipeline всегда пишет camera ack, чтобы можно было проверить, применился ли request.

## Hooks.lua
Работает через DCS Hooks API.

Главные задачи:
- ставить DCS на паузу для pause-first capture;
- делать screenshot через `DCS.makeScreenShot`;
- писать `cv_screenshot_ack.json`;
- писать `cv_pause_ack.json`;
- быть владельцем bbox cache: собирать `unit:getDesc().box` и писать `cv_bbox_cache.txt`;
- следить за simulation start/stop.

Почему screenshot в Hook, а telemetry в Export: DCS разделяет доступные API. Telemetry удобнее в Export API, screenshot делается через Hook callback.

## BBox Cache Contract
Текущий рабочий источник geometry bbox - `Hooks.lua`.

Flow:
- `Hooks.lua` собирает `desc.box` из DCS и пишет `Saved Games/DCS/Logs/cv_bbox_cache.txt`.
- `Export.lua` читает `cv_bbox_cache.txt`.
- Snapshot получает `bbox_min_local`, `bbox_max_local`, `world_basis_raw.axis_x/y/z`, `has_bbox`, `geometry_source`.
- Python проецирует это через `ObjectLocalBBox` + `project_local_bbox`.

`mission_bbox_script.lua` больше не является обязательным рабочим источником bbox. Если рядом есть `mission_bbox_script.lua.bak`, это архивный diagnostic reference, а не файл, от которого должна зависеть генерация миссий.

## File Protocol
Python -> DCS:
- `cv_capture_request.json`
- `cv_camera_request.json`

DCS -> Python:
- `cv_camera_ack.json`
- `cv_screenshot_ack.json`
- `cv_pause_ack.json`
- `cv_snapshots/snapshot_<token>.json`
- `cv_bbox_cache.txt`

Logs:
- `cv_export.log`
- `cv_hook.log`

## Почему Используется Token Sync
Telemetry и screenshot создаются разными callbacks. Token гарантирует, что Python не смешает screenshot одной попытки с snapshot другой.

## Pause-First Sync
Для строгого capture Python сначала пишет `cv_capture_request.json` со статусом `pause_pending`.

Порядок:
- `Hooks.lua` видит `pause_pending`, вызывает `DCS.setPause(true)` и пишет `cv_pause_ack.json`.
- Python ждет `cv_pause_ack.json` с тем же `frame_token`.
- Python переводит тот же request в `pending`.
- `Hooks.lua` вызывает `CV_DATASET_CAPTURE_NOW(...)` в Export environment через `net.dostring_in("export", ...)`.
- `Export.lua` пишет snapshot в уже paused state.
- `Hooks.lua` делает screenshot после появления snapshot, пишет screenshot ack и вызывает `DCS.setPause(false)`.

Это важно: `Export.lua` игнорирует `pause_pending`, поэтому snapshot не должен создаваться до подтвержденной паузы.

## Где Скрипты Должны Лежать После Install
После запуска pipeline:
- `Saved Games/DCS/Scripts/cv_dataset_export.lua`
- `Saved Games/DCS/Scripts/Hooks/cv_dataset_hook.lua`
- `Saved Games/DCS/Scripts/Export.lua` содержит include строки для `cv_dataset_export.lua`

## Частые Проблемы
- Нет screenshot: проверить `cv_hook.log`.
- Нет snapshot: проверить `cv_export.log`.
- Camera не меняется: проверить `cv_camera_ack.json` и поддержку `LoSetCameraPosition` в текущей DCS build.
- Запускается старая миссия: включить `kill_existing_dcs_before_launch` в config.
