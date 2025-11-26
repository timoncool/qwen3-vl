# SuperCaption Qwen3-VL

**Генератор описаний и тегов для фото и видео на базе Qwen3-VL**

Портативное приложение с веб-интерфейсом для работы с мультимодальными моделями Qwen3-VL. Поддерживает Abliterated модели для работы с любым контентом без цензуры.

[![Telegram](https://img.shields.io/badge/Telegram-НЕЙРО--СОФТ-blue?logo=telegram)](https://t.me/neuroport)

---

## Возможности

- **OCR** — распознавание текста с изображений
- **Описание изображений** — генерация описаний в разных стилях (формальный, креативный, SEO и др.)
- **Анализ видео** — описание видеоконтента по кадрам
- **Пакетная обработка** — обработка множества файлов одновременно
- **Сравнение изображений** — анализ нескольких изображений
- **Решение задач** — математические задачи и логические вопросы
- **Object Detection** — обнаружение и локализация объектов на изображениях
- **Thinking Mode** — режим рассуждений модели (Chain-of-Thought)
- **Пресеты промптов** — сохранение и загрузка часто используемых промптов
- **Мультиязычность** — русский и английский интерфейс
- **4-bit квантизация** — экономия видеопамяти (~75% меньше VRAM)

---

## Скриншоты

### OCR — распознавание текста
![OCR](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/01-ocr-text-recognition.png?raw=true)

### Описание изображений
![Description](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/02-image-description.png?raw=true)

### Анализ видео
![Video](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/03-video-analysis.png?raw=true)

### Пакетная обработка
![Batch](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/04-batch-processing.png?raw=true)

### Сравнение нескольких изображений
![Compare](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/05-multi-image-compare.png?raw=true)

### Решение математических задач
![Math](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/06-math-solver.png?raw=true)

### Object Detection — обнаружение объектов
![Detection](https://github.com/timoncool/qwen3-vl/blob/main/screenshots/07-object-detection.png?raw=true)

---

## Доступные модели

### Abliterated (без цензуры)
| Модель | Размер | VRAM (4-bit) |
|--------|--------|--------------|
| Huihui-Qwen3-VL-2B-Instruct-abliterated | 2B | ~2 GB |
| Huihui-Qwen3-VL-4B-Instruct-abliterated | 4B | ~4 GB |
| Huihui-Qwen3-VL-8B-Instruct-abliterated | 8B | ~6 GB |
| Huihui-Qwen3-VL-32B-Instruct-abliterated | 32B | ~20 GB |

### Оригинальные Qwen
| Модель | Размер | VRAM (4-bit) |
|--------|--------|--------------|
| Qwen3-VL-2B-Instruct | 2B | ~2 GB |
| Qwen3-VL-4B-Instruct | 4B | ~4 GB |
| Qwen3-VL-8B-Instruct | 8B | ~6 GB |

---

## Установка

### Windows (рекомендуется)

1. Скачайте и распакуйте архив
2. Запустите `install.bat` для установки зависимостей
3. Запустите `run.bat` для запуска приложения

### Ручная установка

```bash
# Клонирование репозитория
git clone https://github.com/timoncool/qwen3-vl.git
cd qwen3-vl

# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/Mac)
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск
python app.py
```

Приложение запустится на `http://localhost:7860`

---

## Требования

- **Python** 3.10+
- **CUDA** совместимая видеокарта (NVIDIA)
- **VRAM**: минимум 4 GB (для 2B модели с 4-bit квантизацией)
- **RAM**: 16 GB+ рекомендуется

---

## Параметры генерации

| Параметр | Описание |
|----------|----------|
| **Температура** | Креативность (0.1 — точные ответы, 1.0+ — креативные) |
| **Max tokens** | Максимальная длина ответа |
| **Top-p** | Nucleus sampling |
| **Top-k** | Ограничение по количеству токенов |
| **Seed** | Для воспроизводимости результатов |

---

## Структура проекта

```
qwen3-vl/
├── app.py              # Основное приложение
├── install.bat         # Установщик для Windows
├── run.bat             # Запуск приложения
├── run_with_update.bat # Запуск с обновлением
├── requirements.txt    # Зависимости Python
├── screenshots/        # Скриншоты интерфейса
└── README.md
```

---

## Устранение проблем

### CUDA out of memory
- Используйте модель меньшего размера (2B)
- Включите 4-bit квантизацию
- Закройте другие приложения использующие GPU

### Модель не загружается
- Проверьте подключение к интернету
- Убедитесь что достаточно места на диске
- Модели кэшируются в `~/.cache/huggingface/`

---

## Credits

**Портативная версия:**
- [Nerual Dreming](https://t.me/nerual_dreming)
- [Slait](https://t.me/ruweb24)

**Telegram канал:** [НЕЙРО-СОФТ](https://t.me/neuroport)

---

## Лицензия

Проект использует модели [Qwen](https://github.com/QwenLM/Qwen2.5-VL) под лицензией Apache 2.0.

