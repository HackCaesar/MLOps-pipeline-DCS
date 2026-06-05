# scenario_builder

Каталог создает `.miz` миссии через `pydcs`.

## create_mvp_mission.py
Минимальный генератор MVP миссии.

Режимы:
- `ship_only` - один корабль.
- `mixed` - корабль, вертолет, самолет на более свободной дистанции.
- `mixed_close` - корабль, вертолет, самолет ближе друг к другу для проверки overlap/сложных сцен.

Пример:
```powershell
D:\Ai_copy\.venv\Scripts\python.exe D:\Ai_copy\DCS_AutoDataset\scenario_builder\create_mvp_mission.py `
  --output D:\Ai_copy\DCS_AutoDataset\missions\mvp_mixed_close.miz `
  --mode mixed_close
```

Почему этот файл простой: он нужен как стабильная минимальная сцена для проверки end-to-end pipeline. Чем меньше переменных в MVP, тем проще debug sync, camera и bbox.

## create_bbox_test_missions.py
Генерирует 5 regression-test миссий для проверки bbox pipeline.

Миссии:
- `bbox_test_01_clear_mixed.miz` - clear weather, mixed objects.
- `bbox_test_02_cloudy_ship_heavy.miz` - cloudy, несколько кораблей.
- `bbox_test_03_haze_air_heavy.miz` - haze, больше воздушных объектов.
- `bbox_test_04_rain_close_overlap.miz` - rain, близкие/overlap сцены.
- `bbox_test_05_sunset_extreme_angles.miz` - sunset, разные углы.

Запуск:
```powershell
D:\Ai_copy\.venv\Scripts\python.exe D:\Ai_copy\DCS_AutoDataset\scenario_builder\create_bbox_test_missions.py `
  --output-dir D:\Ai_copy\DCS_AutoDataset\missions\bbox_tests
```

Почему именно 5 миссий: это быстрый regression набор. Он проверяет не только один удачный ракурс, а разные комбинации:
- разные heading кораблей;
- разные классы кораблей;
- разные классы самолетов и вертолетов;
- clear/cloudy/haze/rain/sunset;
- close и far layouts.

## BBox В Миссиях
Генераторы миссий больше не встраивают `dcs_export/mission_bbox_script.lua`.

Текущий bbox flow не зависит от mission trigger scripts:
- `Hooks.lua` в DCS runtime собирает `unit:getDesc().box`;
- `Hooks.lua` пишет `cv_bbox_cache.txt`;
- `Export.lua` читает cache и добавляет `bbox_min_local` / `bbox_max_local` в snapshot.

Если в репозитории есть `mission_bbox_script.lua.bak`, это только архивный diagnostic reference. Генерация `.miz` не должна падать, если активного `mission_bbox_script.lua` нет.

## Замечания По pydcs
При генерации могут появляться warnings про livery zip, например `Could not parse livery definition`. Если `.miz` создан, это обычно не критично: pydcs не смог прочитать одну livery, но миссия сохраняется.

## Как Добавить Новую Тестовую Миссию
1. Открыть `create_bbox_test_missions.py`.
2. Добавить новый `MissionSpec` в `MISSIONS`.
3. Использовать классы из `dcs.ships`, `dcs.planes`, `dcs.helicopters`.
4. Проверить, что class inference в `orchestrator/main.py` узнает type name.
5. Если новый тип неизвестен, сначала убедиться, что `Hooks.lua` пишет для него `desc.box` в `cv_bbox_cache.txt`; `projection/model_dims.json` добавлять только как fallback.

## Почему Координаты Около `BASE_X/BASE_Y`
Эта точка находится над морем на Caucasus map. Sea-only сцены проще для bbox и соответствуют задаче корабельной камеры.
