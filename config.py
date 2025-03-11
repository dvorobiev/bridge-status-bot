import os
from dotenv import load_dotenv
from pathlib import Path

# Загрузка переменных окружения
load_dotenv()

# Настройки модели
MODEL_PATH = MODELS_DIR / "best.pt"  # Новый путь к основной модели
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.4))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.3))

# Настройки видео
VIDEO_FPS = 30
VIDEO_FOURCC = "mp4v"

# Пути к директориям
BASE_DIR = Path(__file__).parent
BRIDGE_DETECTOR_DIR = BASE_DIR / "bridge_detector_v2"

# Основные директории проекта
MODELS_DIR = BRIDGE_DETECTOR_DIR / "models"          # Все модели только здесь
OUTPUT_DIR = BRIDGE_DETECTOR_DIR / "output"          # Все выходные данные здесь
DATASET_DIR = BRIDGE_DETECTOR_DIR / "dataset"        # Все данные датасета здесь
TEMP_DIR = BRIDGE_DETECTOR_DIR / "temp"             # Временные файлы внутри проекта

# Структура датасета
DATASET_STRUCTURE = {
    "train": DATASET_DIR / "train",
    "val": DATASET_DIR / "val",
    "new_data": DATASET_DIR / "new_data"
}

# Создание необходимых директорий при импорте
for dir_path in [MODELS_DIR, OUTPUT_DIR, TEMP_DIR] + list(DATASET_STRUCTURE.values()):
    dir_path.mkdir(parents=True, exist_ok=True)

# Классы для детекции
CLASSES = [
    "bridge",           # Мост
    "unknown_class_1",  # Неизвестный класс 1
    "unknown_class_2",  # Неизвестный класс 2
    "unknown_class_3",  # Неизвестный класс 3
    "unknown_class_4",  # Неизвестный класс 4
    "unknown_class_5",  # Неизвестный класс 5
]

# Цвета для визуализации классов
CLASS_COLORS = {
    "bridge": (0, 255, 0),      # Зеленый для мостов
    "unknown_class_1": (255, 0, 0),  # Красный
    "unknown_class_2": (0, 0, 255),  # Синий
    "unknown_class_3": (255, 255, 0),  # Желтый
    "unknown_class_4": (255, 0, 255),  # Пурпурный
    "unknown_class_5": (0, 255, 255),  # Голубой
}

# Конфигурация бота
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CAMERA_URL = os.getenv('CAMERA_URL')

# Параметры детекции
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.3

# Удаляем устаревшие определения путей
# SCREENSHOTS_DIR = "screenshots"  # Удалено
# MODEL_PATH = "models/best.pt"   # Заменено на полный путь
# MODEL_PATH = MODELS_DIR / "best.pt"  # Новый путь к основной модели
# ... existing code ... 