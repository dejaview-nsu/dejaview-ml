# dejaview-ml

Эмбеддинги запросов и фильмов, векторный поиск, индексация каталога.

## Зона ответственности

- Эмбеддинги по модальностям: текст, изображение, кадры видео, аудио
- Нарезка видео на кадры и сегменты, аудио на чанки
- Векторный поиск в Qdrant и замеры качества выдачи по каждому режиму
- Подготовка и обновление индекса каталога
- Офлайн-скрипты: сбор каталога из TMDB (`scripts/catalog/`), индексация фильма (`scripts/index/`)

## Стек

- Python, HuggingFace
- Qdrant, четыре коллекции по модальностям
- Модели и размерности заданы требованием, таблица в `dejaview-docs/architecture.md`
- Веса моделей кэшируются между перезапусками и в репозиторий не коммитятся

## Ссылки

- Задачи и требования: https://ai.nsu.ru/projects/dejaview
- Архитектура, контракты, соглашения: https://github.com/dejaview-nsu/dejaview-docs
- Организация: https://github.com/dejaview-nsu

## Как работаем

- Ветка от `main`: `feat/<номер задачи>-<кратко>`, нейминг в `conventions.md`
- Изменения через Pull Request с ревью, прямой push в `main` закрыт
- Ревьюер назначается автоматически по CODEOWNERS
- Время трекается в Redmine, в задачу, а не в требование

## Запуск

Сервису нужен Qdrant.
При первом старте скачиваются веса моделей (несколько ГБ). Веса кэшируются и при перезапуске повторно не скачиваются.

### Через Docker Compose

Сервис и Qdrant поднимаются в контейнерах.

```bash
cp .env.example .env   # необязательно: без .env берутся значения по умолчанию из compose.yaml
docker compose -f compose.yaml -f compose.local.yaml up --build
```

Чтобы включать и выключать возможность `image` и `video`, `text` и `audio`, например, для более быстрой проверки, задай `ML_TEXT_ENABLED=true` и другие в `.env` или в окружении.

Веса лежат в volume `model-cache`, данные Qdrant в `qdrant-data`.
`docker compose down` их сохраняет, `docker compose down -v` удаляет.

### Локально

Сервис запускается из venv, Qdrant в Docker.
Нужны Python 3.11–3.14 и `ffmpeg` в `PATH` для видео и аудио (на macOS: `brew install ffmpeg`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
docker compose -f compose.yaml -f compose.local.yaml up -d qdrant
uvicorn dejaview_ml.app:app --host 127.0.0.1 --port 8000
```

Настройки читаются из `.env` и переменных окружения с префиксом `ML_`, полный список в `.env.example`.
Модальности включаются и выключаются так же, через `ML_<MODALITY>_ENABLED`.
`ML_DEVICE` принимает `auto`, `cpu`, `cuda` или `mps` (Apple Silicon).
Веса кэшируются в `ML_CACHE_DIR`, по умолчанию `.cache/huggingface`.

### Проверка

```bash
curl -s localhost:8000/health
# {"status":"ok","models":{...},"qdrant":"ok"}, когда все включённые модели в состоянии ready

curl -s -X POST localhost:8000/v1/search/image -H 'Content-Type: image/png' --data-binary @poster.png
curl -s -X POST localhost:8000/v1/search/video -H 'Content-Type: video/mp4' --data-binary @clip.mp4
curl -s -X POST localhost:8000/v1/search/audio -H 'Content-Type: audio/wav' --data-binary @sample.wav
curl -s -X POST localhost:8000/v1/search/text  -H 'Content-Type: application/json' -d '{"query":"фильм про сны"}'
# {"movie_ids":[...]}
```

На пустых коллекциях поиск вернёт `{"movie_ids":[]}`: сервис при старте создаёт коллекции, но не наполняет их. Полный контракт API лежит в [`ml-service.openapi.yaml`](ml-service.openapi.yaml),
в запущенном сервисе Swagger UI доступен на `/docs`.

## Тесты

Обычные тесты работают на заглушках: не скачивают веса и не требуют Qdrant.
Интеграционные тесты по умолчанию пропускаются и включаются переменными окружения:

| Переменная | Что проверяет | Что нужно |
|---|---|---|
| `QDRANT_TEST_URL` | создание коллекций и индексов в настоящем Qdrant | запущенный Qdrant |
| `RUN_MODEL_TESTS=1` | настоящие модели image и video, поиск end-to-end | скачивание весов, `ffmpeg` |
| `RUN_OPTIONAL_MODEL_TESTS=1` | настоящие модели text и audio, поиск end-to-end | скачивание весов, `ffmpeg` |

Тестам с моделями отдельный Qdrant не нужен, они используют встроенный in-memory.

### Локально

Нужно окружение из раздела [«Запуск → Локально»](#локально).

```bash
ruff check src tests
pytest -q
```

Тесты видео пропускаются, если `ffmpeg` нет в `PATH`.

Интеграционные:

```bash
docker compose -f compose.yaml -f compose.local.yaml up -d qdrant
QDRANT_TEST_URL=http://localhost:6333 pytest -q tests/test_qdrant_server.py

RUN_MODEL_TESTS=1 pytest -q tests/test_real_models.py
RUN_OPTIONAL_MODEL_TESTS=1 pytest -q tests/test_optional_real_models.py
```

Веса скачиваются в `ML_CACHE_DIR` (по умолчанию `.cache/huggingface`), поэтому долгим будет только первый запуск.

### В Docker

Тесты собираются в отдельный стейдж `test` из `Dockerfile`, Python и `ffmpeg` локально не нужны.
Флаги `--build-arg` собирают CPU-образ вместо CUDA:

```bash
docker build --target test -t dejaview-ml-test \
  --build-arg BASE_IMAGE=ubuntu:24.04 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .

docker run --rm dejaview-ml-test
```

Образ содержит копию кода, поэтому после изменений его нужно пересобрать.

Интеграционные тесты в Docker. Для Qdrant контейнер подключается к сети compose, где Qdrant доступен
по имени `qdrant`. Для моделей подключается volume с кэшем весов, общий с сервисом:

```bash
docker compose -f compose.yaml -f compose.local.yaml up -d qdrant
docker run --rm --network dejaview-ml_default \
  -e QDRANT_TEST_URL=http://qdrant:6333 \
  dejaview-ml-test python -m pytest -q tests/test_qdrant_server.py

docker run --rm -v dejaview-ml_model-cache:/models \
  -e RUN_MODEL_TESTS=1 -e RUN_OPTIONAL_MODEL_TESTS=1 \
  dejaview-ml-test python -m pytest -q tests/test_real_models.py tests/test_optional_real_models.py
```

## Скрипты

- [`scripts/catalog/`](scripts/catalog/README.md) - сбор начального каталога фильмов из TMDB:
  отбирает фильмы, забирает их метаданные и сохраняет в JSON-файл.
- [`scripts/index/`](scripts/index/README.md) - индексация фильма: видеофайл и `movie_id` на входе,
  эмбеддинги в Qdrant и строка в `movies` на выходе, после чего фильм находится поиском.
