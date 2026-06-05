# validators

Каталог отвечает за качество bbox. После Step 4 декомпозиции структура такая:

| Файл | Роль |
|---|---|
| `visible_bbox_refiner.py` | Фасадный класс `VisibleBBoxRefiner(AirborneRefinerMixin, ShipRefinerMixin)`. Публичный API: `refine_frame(image_path, frame_metadata)`, `refine_object(image, bbox, class_name, debug_context)`. Содержит общие helpers: `_roi_for_bbox`, `_bbox_from_components`, `_pad_bbox`, `_apply_refined_bbox`, `_refresh_frame_stats`, `_safe_debug_token`, `_draw_debug_bbox`, `_components_intersecting_bbox`, `_components_debug_union_bbox`, `_limit_component_debug`. |
| `airborne_refiner.py` | `AirborneRefinerMixin` — refinement для `airplanes` / `helicopters`. LAB + HSV + Canny → connected components → geometry-overlap filtering. Включает все `_refine_airborne`, `_select_airborne_*`, `_airborne_*`, `_record_airborne_*`, `_write_airborne_debug_artifacts`. |
| `ship_refiner.py` | `ShipRefinerMixin` — refinement для `ships`. Edge-projection основной путь + grabcut fallback. Включает все `_refine_ship*`, `_ship_*`, `_reject_ship_*`, `_split_ship_*`, `_clamp_ship_*`, `_write_ship_*`, `_diagnose_components`, `_split_components_by_bbox`, `_component_diagnostic`. |
| `refined_bbox.py` | Общий `@dataclass RefinedBBox` (импортируется из всех трёх). |
| `bbox_geometry.py` | Pure 2D bbox helpers: `clip_bbox`, `bbox_area`, `bbox_overlap_area`, `bbox_center`. |
| `components.py` | Wrappers вокруг `cv2.connectedComponentsWithStats`: `connected_components`, `component_rows_cols`. |
| `refine_dataset.py` | Batch-CLI для пересчёта уже сгенерированного датасета. |
| `diagnose_bbox_axis_mapping.py` | Offline-диагностика axis-mapping (не production). |

## Mixin Pattern: почему так

`AirborneRefinerMixin` и `ShipRefinerMixin` опираются на shared helpers (`self._roi_for_bbox`, `self._pad_bbox`, …) и configuration (`self.config`, `self.ship_padding_px`, …), которые приходят из фасада `VisibleBBoxRefiner`. Mixin не самодостаточен — он MUST использоваться через `VisibleBBoxRefiner(AirborneRefinerMixin, ShipRefinerMixin)`.

Это позволяет физически разделить ~1500 строк airborne/ship-логики по файлам, **сохранив все вызовы `self.X()` в исходной форме**. Никаких алгоритмических изменений; регрессия зелёная байт-в-байт.

## Что нельзя менять без Algorithm Change Proposal

- Любые thresholds (`color_distance > 18.0`, `value < 115`, `selected_height < max(6.0, 0.12 * geometry_height)`, и т.п.) — менять их означает менять качество разметки.
- `filter_reject_reasons` строки (например, `"rejected_zero_geometry_overlap"`, `"airborne_rejected_large_roi_fraction"`) — они идут в `visible_bbox_refinement.airborne_reject_reasons` metadata. Изменение даже одной строки = diff в golden output.
- Порядок проверок в `_split_ship_components_conservative` — он определяет какой `reject_reason` записывается первым.
- Имена debug-файлов (`<frame>__<id>__<class>__<type>__<method>__overlay.png` и т.п.) — они зашиты в `visible_bbox_refinement.*_debug_artifacts.*_png` и регрессия их сравнивает (нормализуя tmpdir-префикс).

## visible_bbox_refiner.py
Этот модуль заменяет грубый geometry bbox на bbox видимых пикселей.

### Зачем Он Нужен
Geometry bbox строится в первую очередь из DCS local bbox (`bbox_min_local` / `bbox_max_local`) и только fallback-ом из `ObjectCuboid` + `model_dims.json`. У кораблей он часто слишком большой: включает пустое небо вокруг мачт, воду, объем модели, который не соответствует силуэту. У самолетов и вертолетов 3D bbox зависит от ориентации и тоже дает лишние пустоты.

`VisibleBBoxRefiner` использует geometry bbox только как область поиска. Финальный bbox строится по screenshot.

### Основной Flow
1. `refine_frame(image_path, frame_metadata)` читает PNG кадра.
2. Для каждого объекта берет geometry proposal из `bbox_xyxy_geometry_px` или текущего `bbox_xyxy_px`.
3. Вызывает `refine_object`.
4. Если объект уточнен, вызывает `_apply_refined_bbox`.
5. Если объект не найден надежно, помечает его `reject`.
6. Обновляет `object_stats`, `usable`, `invalid_reasons`.

### Почему Объекты Лучше Reject, Чем Плохой BBox
Если refinement не уверен, экспортировать bbox нельзя. Плохая bbox-разметка вреднее, чем отсутствие объекта: detector начнет учиться на неправильных границах. Поэтому объект получает:
```json
"quality_tier": "reject",
"validation": {"valid": false},
"projection": {"bbox_xyxy_px": null}
```

YOLO/COCO exporter такие объекты пропускает.

### Airborne Refinement
Для `airplanes`, `helicopters` используется `_refine_airborne`.

Метод:
- берет ROI вокруг geometry bbox;
- оценивает фон по border pixels;
- ищет отличие объекта от фона через LAB color distance, HSV saturation/value и Canny edges;
- собирает connected components;
- выбирает компоненты рядом с ожидаемым центром;
- отбрасывает компоненты, которые слишком большие или слишком маленькие относительно geometry proposal.

Почему так: самолеты и вертолеты обычно находятся на фоне неба/воды и отличаются по цвету/краям. Но рядом могут быть другие объекты, поэтому есть фильтры по размеру и расстоянию.

### Ship Refinement
Для `ships` используется `_refine_ship`.

Основной путь:
- `_refine_ship_from_edges`
- edge projection внутри ограниченной области вокруг geometry bbox

Fallback:
- `_refine_ship_from_grabcut`

Почему нужны ограничения: корабль может иметь высокую мачту, корпус, отражения и соседние самолеты/вертолеты. Если разрешить edge detector искать слишком широко, bbox может захватить чужой объект или пустое небо. Поэтому добавлен `_ship_bbox_plausible`, который запрещает слишком маленькие или слишком далеко ушедшие bbox.

### Почему Ship BBox Иногда Включает Пустое Небо
Разметка bbox axis-aligned. Если у корабля есть высокая мачта и низкий длинный корпус, прямоугольник неизбежно включает пустое пространство между ними. Это не ошибка алгоритма, а свойство прямоугольной bbox-разметки. Pixel-perfect граница потребовала бы segmentation mask.

### Главные Поля, Которые Модуль Добавляет
```json
"projection": {
  "bbox_xyxy_geometry_px": [...],
  "bbox_xyxy_visible_px": [...],
  "bbox_xyxy_px": [...]
},
"quality_tier": "exact_visible",
"confidence_source": "visible_pixels",
"visible_bbox_refinement": {
  "status": "applied",
  "method": "ship_edge_projection"
}
```

## refine_dataset.py
Batch utility для старого датасета. Если dataset уже был сгенерирован до добавления visible refinement, этот скрипт пересчитывает metadata, YOLO, COCO и overlays.

Запуск всего датасета:
```powershell
D:\Ai_copy\.venv\Scripts\python.exe D:\Ai_copy\DCS_AutoDataset\validators\refine_dataset.py `
  --config D:\Ai_copy\DCS_AutoDataset\configs\config.yaml
```

Запуск одного кадра:
```powershell
D:\Ai_copy\.venv\Scripts\python.exe D:\Ai_copy\DCS_AutoDataset\validators\refine_dataset.py `
  --config D:\Ai_copy\DCS_AutoDataset\configs\config.yaml `
  --frame-id frame_id_here
```

Важно: после обработки одного кадра COCO теперь все равно пересобирается по всем metadata, а не только по одному кадру.

## Когда Использовать Какой Путь
- Новый capture через `orchestrator/main.py`: refinement уже встроен, вручную запускать `refine_dataset.py` не нужно.
- Старые изображения/metadata: использовать `refine_dataset.py`.

## Ограничения
- Требуется OpenCV (`opencv-python`).
- Это image heuristic, не engine-side instance mask.
- Слишком маленькие, частично невидимые или сомнительные объекты могут быть rejected.
