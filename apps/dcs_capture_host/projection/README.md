# projection

Каталог отвечает за геометрию: перевод координат DCS world space в pixel space.

## projector.py
Главный файл с математикой projection.

## Coordinate Conventions
DCS world:
- `X` - north
- `Y` - up
- `Z` - east
- meters

Camera frame для CV:
- `x_c` - right
- `y_c` - up
- `z_c` - forward

Pixel projection:
```text
u = fx * (x_c / z_c) + cx
v = cy - fy * (y_c / z_c)
```

Минус в `v` нужен потому, что image coordinates растут вниз, а camera `y_c` растет вверх.

## CameraIntrinsics
Хранит параметры камеры:
- image width/height;
- `fx`, `fy`;
- optical center `cx`, `cy`;
- near/far planes;
- источник FOV.

Сейчас intrinsics строятся из vertical FOV:
```python
CameraIntrinsics.from_vertical_fov(...)
```

Почему `fx = fy`: предполагаются square pixels. Horizontal FOV получается автоматически из ширины изображения.

## CameraPose
Хранит camera basis:
- `position_w`
- `forward_w`
- `up_w`
- `right_w`

Метод `world_to_camera` делает:
```text
p_rel = p_world - camera_position
x_c = dot(right_w, p_rel)
y_c = dot(up_w, p_rel)
z_c = dot(forward_w, p_rel)
```

## ObjectLocalBBox
Основной путь для новых кадров. Описывает DCS local bbox:
- `origin_w` - world position объекта;
- `axis_x_w` - forward из `world_basis_raw.axis_x`;
- `axis_y_w` - up из `world_basis_raw.axis_y`;
- `axis_z_w` - right из `world_basis_raw.axis_z`;
- `bbox_min_local` / `bbox_max_local` из snapshot.

`project_local_bbox` проецирует 8 углов этого local bbox в пиксели и возвращает geometry bbox proposal.

## ObjectCuboid
Fallback путь. Используется только если DCS snapshot не содержит `bbox_min_local` / `bbox_max_local`.

Он строит oriented cuboid из catalog dimensions (`model_dims.json`) и basis объекта. Это хуже, чем DCS `desc.box`, но позволяет не падать на объектах без bbox cache.

## ProjectionResult
Результат projection:
- projected 2D corner points;
- clipped bbox;
- unclipped bbox;
- depth range;
- truncated flag;
- reasons if invisible.

## model_dims.json
Fallback каталог размеров объектов.

Он нужен только для fallback, если DCS не дал `desc.box` через `cv_bbox_cache.txt`. Примеры:
- `ships`
- `helicopters`
- `airplanes`
- `uh-1h`
- `a-10c`
- `f-16c_50`

Если добавляешь новый тип техники и видишь `No dimensions for type=...`, добавь размер сюда.

## Почему Projection Не Считается Финальной Разметкой
Geometry bbox не совпадает с видимым силуэтом:
- у кораблей мачта и корпус дают большой прямоугольник;
- у самолетов крылья/хвост зависят от ракурса;
- при частичной видимости 3D bbox может включать невидимые части;
- у разных моделей `desc.box` может быть грубым.

Поэтому `projection` дает стартовую геометрию, а `validators/visible_bbox_refiner.py` делает финальный bbox.
