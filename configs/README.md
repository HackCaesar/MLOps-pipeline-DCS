# configs/

`pipeline.yaml` — главный config, его читают все CLI приложения.
`classes.yaml` — имена классов для detection, согласовано с текущим COCO output DCS_AutoDataset.

Прочие `*.yaml` файлы (`dcs_capture.yaml`, `enrichment.yaml`, `yolox_train.yaml`, и т.д.) появятся в соответствующих фазах. На MVP всё живёт в `pipeline.yaml`.

## `${var}` substitution

Loader (`packages/common/config.py`) поддерживает интерполяцию `${section.key}` со ссылками на другие поля внутри того же файла. Резолвер итеративный — допустимы цепочки и порядок не важен.

Пример:

```yaml
storage:
  root_dir: /workspace/storage
  datasets_dir: ${storage.root_dir}/datasets

data:
  raw_dataset_dir: ${storage.datasets_dir}/raw/dcs_001
```

Распознаются: `${a.b}`, `${a.b.c}` (любая глубина). `${VAR}` без точки — пока не поддерживается (можно расширить через env var fallback позже).

## Locked decisions

Ключевые архитектурные решения (размер входа, multiscale, набор классов и т.д.) зафиксированы непосредственно в конфигах и описаны в [`../docs/architecture.md`](../docs/architecture.md) и [`../docs/data_flow.md`](../docs/data_flow.md).
