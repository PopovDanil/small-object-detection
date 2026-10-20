# Small-object detection on VisDrone — итоговые артефакты

Архив с результатами проекта **P17 Small-Object Detection in Aerial Images**: сравнение
обычного сжатия кадра (640), инференса на повышенном разрешении (1280) и тайлинга
(640×640 в родном разрешении, перекрытие 20 %) для YOLO26m и YOLO26x, дообученных на
VisDrone2019-DET. Код: <https://github.com/PopovDanil/small-object-detection>.

**С чего начать:** `report/results.md` — полный отчёт (таблицы, графики, выводы, разбор
ошибок). Все числа в нём взяты из `experiments/metrics/metrics_all.csv`.

```
report/                       итоговый отчёт
  results.md                  отчёт (открывается в любом markdown-просмотрщике / на GitHub)
  figures/                    картинки, на которые ссылается отчёт
checkpoints/                  финальные веса (лучшая эпоха по val mAP50-95)
  yolo26m_s640/best.pt        YOLO26m, 21.8 M параметров, 44 МБ
  yolo26m_s640/args.yaml      точные аргументы обучения Ultralytics
  yolo26x_s640/best.pt        YOLO26x, 58.8 M параметров, 118 МБ
  yolo26x_s640/args.yaml
experiments/                  всё, что посчитано в экспериментах
  metrics/                    ГЛАВНОЕ: все метрики голыми числами
  conditions/                 сырые результаты каждого прогона оценки
  training/                   логи и кривые обучения, пилот, конфиг
  figures/                    все графики экспериментов
  notebooks/                  выполненные ноутбуки (с выводом ячеек)
README.md                     этот файл
```

## checkpoints/ — как использовать веса

Модели обучены на 10 классах VisDrone (id 0–9: pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor) при `imgsz=640`, но работают на любом входе.

```python
from ultralytics import YOLO  # ultralytics==8.4.175

model = YOLO("checkpoints/yolo26m_s640/best.pt")
model.predict("frame.jpg", imgsz=1280, conf=0.25, max_det=1000)   # лучший режим по отчёту
```

Чтобы посчитать метрики тем же оценщиком, что в отчёте (из репозитория):

```python
from src.evaluation.metrics import evaluate_split, load_split_paths
images, labels = load_split_paths("test")
evaluate_split(labels, images, model, imgsz=1280)                       # 1280
evaluate_split(labels, images, model, inference="tiled", overlap=0.20)  # тайлинг
```

Либо положить папки `yolo26*_s640/` в `runs/finetune/` репозитория — ноутбуки
`highres.ipynb` / `tiling.ipynb` / `results.ipynb` найдут их сами (`EVAL_STAGE=finetune`).

## experiments/metrics/ — все метрики (test-dev, 1 610 кадров, 75 101 объект)

| Файл | Что внутри |
|---|---|
| `metrics_all.csv` | **все метрики, значения 0–1**, строка = модель × режим (`full-640`, `full-1280`, `tiled-ov20`; для yolo26m ещё `tiled-ov00`, `tiled-ov40`) |
| `metrics_by_native_size.csv` | те же метрики отдельно по родному размеру кадра (6 групп: 960×540 … 1920×1080) |
| `metrics_by_native_size_delta.csv` | разница каждого режима с 640 внутри группы кадров |
| `results_tables.md` | итоговые таблицы 1–4 в markdown, значения ×100 |
| `table1_overall.csv` | общие метрики + время, разница с 640 |
| `table2_size_yolo26m.csv`, `table2_size_yolo26x.csv` | по размеру объектов (tiny / small / medium / large) |
| `table3_capacity.csv` | yolo26x − yolo26m в каждом режиме |
| `table4_overlap.csv` | перекрытие тайлов 0 / 20 / 40 % (yolo26m) |

Колонки `metrics_all.csv`:

| Колонка | Смысл |
|---|---|
| `map50_95`, `map50` | COCO mAP при IoU 0.50:0.95 / 0.50 (conf 0.001, ≤1000 рамок на кадр) |
| `recall50_25` | recall при IoU ≥ 0.5 и уверенности ≥ 0.25 |
| `ap_<g>`, `ap50_<g>`, `recall_<g>`, `n_<g>` | то же по группе размеров `<g>` + число объектов |
| `runtime_mean_ms`, `runtime_p95_ms` | мс на исходный кадр: Tesla T4, батч 1, 10 кадров прогрева, CUDA sync; для тайлинга — все тайлы + склейка; первые 200 кадров |

Группы размеров (площадь рамки в пикселях **исходного** кадра):
`tiny` < 16², `small_only` 16²–32², `medium` 32²–96², `large` ≥ 96² — не пересекаются,
в сумме дают все объекты (27 345 + 23 491 + 21 856 + 2 409 = 75 101). `small` (< 32²) —
определение COCO (= tiny + small_only), оставлено для сравнения с литературой.

## experiments/conditions/ — сырые результаты прогонов оценки

| Файл | Прогон |
|---|---|
| `full_640.csv`, `full_1280.csv` | полный кадр на 640 / 1280, обе модели (`highres.ipynb`) |
| `tiled_ov00.csv`, `tiled_ov20.csv`, `tiled_ov40.csv` | тайлинг с перекрытием 0 / 20 / 40 % (`tiling.ipynb`; 0 и 40 % — только yolo26m) |
| `tiling_overlap_sweep_yolo26m.csv` | три значения перекрытия в одной таблице |
| `tiling_vs_baseline.csv` | тайлинг 20 % против 640 (из `tiling.ipynb`) |

## experiments/training/ — обучение

| Файл | Что внутри |
|---|---|
| `training_summary.csv` | эпох, лучшая эпоха, лучший val mAP50–95 / mAP50, батч, время (ч) |
| `training_config.json` | общий конфиг обоих запусков + пиковая память пилота |
| `pilot_result.json`, `pilot.log` | пилот выбора батча (yolo26x, 1 эпоха на 2 % train: батч 4/8/16) |
| `yolo26m_s640/`, `yolo26x_s640/` | `results.csv` (метрики по эпохам на val), `args.yaml`, кривые P/R/F1/PR, confusion matrix, примеры батчей train/val |
| `finetune_val_curves.png` | val mAP по эпохам для обеих моделей |
| `yolo26m.log`, `yolo26x.log` | логи запусков на Kaggle |

> val-метрики обучения считает валидатор Ultralytics (без учёта ignored regions, другой лимит
> рамок) — их нельзя напрямую сравнивать с числами на test-dev.

## experiments/figures/ и experiments/notebooks/

- `figures/`: точность против времени, изменение по размерам, перекрытие тайлов, сетка тайлов,
  три случая разбора ошибок (скопление tiny-объектов, крупный объект, граница тайла).
- `notebooks/`: выполненные `highres`, `tiling`, `results` (оценка) и `finetune_*` (обучение) —
  с выводом всех ячеек, можно открыть и посмотреть без запуска.

## Протокол в двух строках

Дообучение одно (640, 50 эпох, батч 16, seed 42, лучшая эпоха по val); три режима инференса
на **одних и тех же** весах; один оценщик для всех режимов; val — только выбор эпохи, все
числа — test-dev. Один прогон без доверительных интервалов: разницы < ~0.5 пункта — шум.
