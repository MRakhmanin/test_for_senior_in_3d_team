# Сервис нормализации ориентации КТ (Docker + FastAPI)

Этот репозиторий содержит API-сервис на FastAPI, который:
- принимает КТ-исследование головы в формате NIfTI (`.nii` / `.nii.gz`);
- предсказывает углы поворота CNN-моделью;
- формирует матрицу поворота;
- возвращает повёрнутый объём и JSON с результатом.

## 1. Что нужно перед запуском

- Установленный Docker.
- Файл обученной модели (`.pt` checkpoint), например:
  - `artifacts/train_runs/angle_regressor_axis_cv3/fold_0/best.pt`

## 2. Сборка Docker-образа

Из корня проекта:

```bash
docker build -t ct-orientation-api .
```

## 3. Запуск контейнера (CPU)

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_CHECKPOINT=/app/checkpoints/best.pt \
  -e DEVICE=cpu \
  -v /absolute/path/to/checkpoints:/app/checkpoints \
  ct-orientation-api
```

Где:
- `MODEL_CHECKPOINT` — путь к чекпоинту внутри контейнера;
- `/absolute/path/to/checkpoints` — путь на вашей машине, где лежит `best.pt`.

## 4. Запуск контейнера (GPU, если доступна)

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e MODEL_CHECKPOINT=/app/checkpoints/best.pt \
  -e DEVICE=auto \
  -v /absolute/path/to/checkpoints:/app/checkpoints \
  ct-orientation-api
```

Если CUDA доступна внутри контейнера, сервис автоматически выберет `cuda` (при `DEVICE=auto`).

## 5. Проверка работоспособности

### Healthcheck

```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

### Запрос на обработку исследования

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/absolute/path/to/CQ500CT6.nii.gz" \
  --output result.zip
```

## 6. Формат результата

Сервис возвращает ZIP-архив `result.zip`:
- `rotated.nii.gz` — повёрнутый КТ-объём;
- `result.json` — углы и матрица поворота.

Пример `result.json`:

```json
{
  "angles_deg": {
    "roll": -2.13,
    "pitch": 1.47,
    "yaw": -6.82
  },
  "rotation_matrix_3x3": [
    [0.99, -0.11, 0.02],
    [0.11, 0.99, -0.03],
    [-0.02, 0.03, 0.99]
  ]
}
```

## 7. Типовые проблемы

- **`MODEL_CHECKPOINT env var is required`**  
  Не задана переменная `MODEL_CHECKPOINT`.

- **`MODEL_CHECKPOINT not found`**  
  Неправильный путь внутри контейнера или неверный `-v` mount.

- **`Only .nii/.nii.gz files are supported`**  
  Загружен файл другого формата.

- **GPU не используется**  
  Проверьте запуск с `--gpus all` и наличие CUDA runtime для Docker.

