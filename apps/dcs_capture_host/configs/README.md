# configs

Каталог содержит YAML конфиги pipeline.

## config.yaml
Основной конфиг для обычной генерации датасета.

Главные блоки:
- `paths` - пути к DCS, Saved Games, output dataset.
- `pipeline` - количество сцен/кадров, strict sync, allowed quality tiers.
- `generation` - классы объектов, диапазоны, weather presets.
- `camera` - высота камеры, FOV, размер изображения.
- `validation` - базовые правила валидности bbox.
- `bbox_refinement` - настройки visible bbox refinement.
- `export` - форматы output.
- `runtime` - timeouts, launch behavior.
- `integration` - JSON files для Python <-> DCS обмена.

## bbox_test_capture.yaml
Специальный конфиг для regression tests.

Отличия:
- `frames_per_scene: 3`
- `max_scenes: 5`
- `close_dcs_on_exit: true`
- `kill_existing_dcs_before_launch: true`
- `kill_dcs_on_exit: true`

Почему отдельный конфиг: при тесте разных `.miz` нужно гарантированно закрывать DCS между миссиями. Обычный config может оставлять DCS открытым, что удобно для ручной работы, но опасно для batch regression.

## paths
Самый важный блок при переносе на другое устройство:
```yaml
paths:
  dcs_exe: "D:/Steam/steamapps/common/DCSWorld/bin/DCS.exe"
  saved_games_dir: "C:/Users/maest/Saved Games/DCS"
  scripts_dir: "C:/Users/maest/Saved Games/DCS/Scripts"
  screenshots_dir: "C:/Users/maest/Saved Games/DCS/ScreenShots"
  output_root: "./dataset"
```

На новом ПК нужно заменить user name и путь установки DCS.

## allowed_quality_tiers_for_training
```yaml
allowed_quality_tiers_for_training: ["exact_visible", "exact", "strong", "approximate"]
```

Сейчас новые хорошие bbox получают `exact_visible`. Старые tiers оставлены, чтобы exporter мог работать со старой metadata. Если хочешь обучаться только на новых visible bbox, оставь только:
```yaml
allowed_quality_tiers_for_training: ["exact_visible"]
```

## camera
```yaml
camera:
  fixed_height_m: 15.0
  roll_deg: 0.0
  fov:
    vertical_deg: 60.0
```

Высота `15 m` - ключевое требование задачи. Pitch не задается напрямую в config, он получается из camera basis. Orchestrator планирует basis так, чтобы pitch был около `0`.

## bbox_refinement
```yaml
bbox_refinement:
  enabled: true
  ignore_bottom_px: 45
  min_refined_area_px: 16
  airborne_padding_px: 2
  ship_padding_px: 3
```

`ignore_bottom_px` отсекает нижний DCS overlay/HUD area. Padding добавляет небольшой запас к найденному visible bbox.

## runtime
Обычный запуск может держать DCS открытым:
```yaml
close_dcs_on_exit: false
pause_first_capture: true
```

`pause_first_capture: true` включает двухстадийный capture request: `pause_pending` -> `cv_pause_ack.json` -> `pending`. Это нужно, чтобы `Export.lua` снимал координаты только после паузы.

Regression запуск должен закрывать DCS:
```yaml
close_dcs_on_exit: true
kill_existing_dcs_before_launch: true
kill_dcs_on_exit: true
```

## integration
Эти пути должны совпадать с Lua scripts:
```yaml
request_file: "C:/Users/.../Saved Games/DCS/Logs/cv_capture_request.json"
snapshot_dir: "C:/Users/.../Saved Games/DCS/Logs/cv_snapshots"
screenshot_ack_file: "C:/Users/.../Saved Games/DCS/Logs/cv_screenshot_ack.json"
pause_ack_file: "C:/Users/.../Saved Games/DCS/Logs/cv_pause_ack.json"
camera_request_file: "C:/Users/.../Saved Games/DCS/Logs/cv_camera_request.json"
camera_ack_file: "C:/Users/.../Saved Games/DCS/Logs/cv_camera_ack.json"
```
