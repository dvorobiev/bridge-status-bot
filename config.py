import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройки модели
MODEL_PATH = "models/best.pt"  # Путь к локальной модели
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.3

# Настройки видео
VIDEO_FPS = 30
VIDEO_FOURCC = "mp4v"

# Пути к директориям
SCREENSHOTS_DIR = "screenshots"
OUTPUT_DIR = "output"
MODELS_DIR = "models"

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