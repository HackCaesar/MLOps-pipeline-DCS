# exporters

Каталог отвечает за запись training annotations.

## coco_yolo.py
Содержит `DatasetExporter`.

## Что Экспортируется
Из rich metadata exporter пишет:
- YOLO labels в `dataset/yolo_labels/*.txt`.
- COCO annotations в `dataset/annotations/_annotations.coco.json`.

## Почему Exporter Фильтрует Объекты
Exporter не должен экспортировать все объекты из metadata. Он пропускает:
- кадры с `usable: false`;
- объекты с `validation.valid: false`;
- объекты без `projection.bbox_xyxy_px`;
- объекты с quality tier, которого нет в `allowed_quality_tiers_for_training`.

Это сделано потому, что metadata хранит и rejected объекты для анализа, но training annotations должны содержать только надежную разметку.

## YOLO Format
Каждая строка:
```text
class_id x_center_norm y_center_norm width_norm height_norm
```

Координаты нормируются на image width/height.

Если кадр `usable=false`, файл `.txt` создается пустым. Это удобно для совместимости с YOLO tooling.

## COCO Format
COCO содержит:
- `images`
- `annotations`
- `categories`

Annotation bbox пишется в формате:
```json
[x_min, y_min, width, height]
```

Дополнительно сохраняются поля:
- `quality_tier`
- `confidence_source`

## Class IDs
Class IDs берутся из порядка `generation.allowed_classes` в config:
```yaml
allowed_classes: ["ships", "helicopters", "airplanes"]
```

Значит:
- `ships` -> `0`
- `helicopters` -> `1`
- `airplanes` -> `2`

Если порядок классов изменить, ID в YOLO/COCO тоже изменятся.

## Важное Про Single-frame Refinement
`validators/refine_dataset.py --frame-id ...` пересчитывает один metadata/YOLO/overlay, но COCO пересобирается по всем metadata. Это нужно, чтобы `_annotations.coco.json` не превратился случайно в COCO только одного кадра.
