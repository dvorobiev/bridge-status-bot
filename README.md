# Float Bridge Detector

Проект для автоматической детекции мостов на изображениях и видео с использованием локальной модели YOLO.

## Возможности

- Детекция мостов на статических изображениях
- Обработка видео с детекцией мостов в реальном времени
- Визуализация результатов детекции
- Настраиваемые параметры детекции
- Логирование процесса обработки
- Работа с локальной моделью YOLO

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/float_bridge.git
cd float_bridge
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Поместите вашу обученную модель YOLO в директорию `models/` с именем `best.pt`

## Использование

### Обработка изображения

```python
from bridge_detector import BridgeDetector

detector = BridgeDetector()
results = detector.detect_image("path/to/image.jpg")
```

### Обработка видео

```python
from bridge_detector import BridgeDetector

detector = BridgeDetector()
detector.process_video("input_video.mp4", "output_video.mp4")
```

## Конфигурация

Основные настройки находятся в файле `config.py`:

- `MODEL_PATH`: Путь к локальной модели YOLO
- `CONFIDENCE_THRESHOLD`: Порог уверенности для детекции (по умолчанию 0.4)
- `IOU_THRESHOLD`: Порог IoU для подавления дублирующихся детекций (по умолчанию 0.3)
- `VIDEO_FPS`: Частота кадров для выходного видео
- `VIDEO_FOURCC`: Кодек для кодирования видео
- `CLASSES`: Список классов для детекции
- `CLASS_COLORS`: Цвета для визуализации разных классов

## Структура проекта

```
float_bridge/
├── bridge_detector.py    # Основной класс для детекции
├── config.py            # Конфигурационные параметры
├── requirements.txt     # Зависимости проекта
├── models/             # Директория для моделей
│   └── best.pt        # Обученная модель YOLO
├── screenshots/        # Директория для временных изображений
└── output/            # Директория для результатов
```

## Обучение модели

Для обучения модели YOLO на своих данных:

1. Подготовьте датасет в формате YOLO
2. Создайте файл конфигурации для обучения
3. Запустите обучение:
```bash
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

## Лицензия

MIT 