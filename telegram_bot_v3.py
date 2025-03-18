import os
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import random
import logging
import torch
import yaml
from typing import Optional, Dict, List
from logging.handlers import RotatingFileHandler
import time
from functools import wraps
import traceback

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError
from ultralytics import YOLO
from loguru import logger

from config import (
    TELEGRAM_TOKEN,
    CAMERA_URL,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    BASE_DIR,
    DATASET_DIR,
    MODELS_DIR,
    TEMP_DIR,
    OUTPUT_DIR
)

class FileSystemManager:
    """Менеджер для работы с файловой системой"""
    @staticmethod
    def ensure_directories(*paths: Path):
        """Создание нескольких директорий"""
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана директория: {path}")

    @staticmethod
    def setup_logging(log_path: Path, max_size_mb: int = 10, backup_count: int = 5):
        """Настройка логирования с ротацией файлов"""
        try:
            # Используем относительный путь в текущей директории
            log_dir = Path.cwd() / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / 'bot.log'
            
            # Проверяем права доступа
            if log_dir.exists():
                os.chmod(log_dir, 0o755)  # Устанавливаем права на директорию
                if log_file.exists():
                    os.chmod(log_file, 0o644)  # Устанавливаем права на файл
            
            # Настройка форматирования
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            formatter = logging.Formatter(log_format)
            
            # Настройка ротации файлов
            file_handler = RotatingFileHandler(
                str(log_file),
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            
            # Настройка вывода в консоль
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Настройка корневого логгера
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
            # Отключение лишних логов
            logging.getLogger('httpx').setLevel(logging.WARNING)
            
        except Exception as e:
            print(f"Ошибка при настройке логирования: {str(e)}")
            print("Продолжаем работу только с выводом в консоль")
            
            # Настраиваем только консольное логирование
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.addHandler(console_handler)

class ModelManager:
    """Менеджер для работы с моделями YOLO"""
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.current_model: Optional[YOLO] = None
        
    def load_model(self, model_path: Optional[Path] = None) -> YOLO:
        """Загрузка модели с обработкой ошибок"""
        try:
            if model_path is None:
                model_path = self.models_dir / "best.pt"
            
            if model_path.exists():
                logger.info(f"Загрузка существующей модели: {model_path}")
                self.current_model = YOLO(str(model_path))
            else:
                logger.info("Создание новой модели YOLOv8n")
                self.current_model = YOLO('yolov8n.pt')
            
            return self.current_model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise

class BridgeDetectorV3:
    """Улучшенная версия детектора мостов"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация BridgeDetectorV3")
        
        # Инициализация менеджеров
        self.fs_manager = FileSystemManager()
        self.model_manager = ModelManager(MODELS_DIR)
        
        # Создание необходимых директорий
        self.fs_manager.ensure_directories(
            DATASET_DIR,
            MODELS_DIR,
            TEMP_DIR,
            OUTPUT_DIR
        )
        
        # Инициализация модели
        self.model = self.model_manager.load_model()
        self.camera_url = CAMERA_URL
        self.training_data: Dict[str, dict] = {}
        self.is_training_mode = False
        
        # Директория для новых данных
        self.new_data_dir = DATASET_DIR / "new_data"
        self.fs_manager.ensure_directories(self.new_data_dir)

    async def capture_frame(self) -> Optional[np.ndarray]:
        """Захват кадра с камеры с обработкой ошибок"""
        try:
            cap = cv2.VideoCapture(self.camera_url)
            if not cap.isOpened():
                raise RuntimeError("Не удалось подключиться к камере")
            
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Не удалось получить кадр")
            
            return frame
        except Exception as e:
            self.logger.error(f"Ошибка при захвате кадра: {str(e)}")
            return None
        finally:
            if 'cap' in locals():
                cap.release()

    def _save_frame(self, frame: np.ndarray, label: str) -> Optional[Path]:
        """Сохранение кадра для обучения с валидацией"""
        try:
            if frame is None:
                raise ValueError("Получен пустой кадр")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self.new_data_dir if not self.is_training_mode else DATASET_DIR / "train"
            self.fs_manager.ensure_directories(save_dir)
            
            image_path = save_dir / f"bridge_{timestamp}.jpg"
            label_path = save_dir / f"bridge_{timestamp}.txt"
            
            # Проверка и сохранение изображения
            if not cv2.imwrite(str(image_path), frame):
                raise IOError("Не удалось сохранить изображение")
            
            # Валидация размеров изображения
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                raise ValueError("Некорректные размеры изображения")
            
            # Создание аннотации
            class_id = 0 if label == 'closed' else 1
            annotation = f"{class_id} 0.5 0.5 0.8 0.8"
            
            # Сохранение аннотации
            with open(label_path, 'w') as f:
                f.write(annotation)
            
            return image_path
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении кадра: {str(e)}\n{traceback.format_exc()}")
            return None

    def _prepare_training_data(self) -> bool:
        """Подготовка данных для обучения с валидацией"""
        try:
            # Проверка наличия новых данных
            image_files = list(self.new_data_dir.glob("*.jpg"))
            if not image_files:
                self.logger.warning("Нет новых данных для обучения")
                return False
            
            # Создание директорий
            train_dir = DATASET_DIR / "train"
            val_dir = DATASET_DIR / "val"
            self.fs_manager.ensure_directories(train_dir, val_dir)
            
            # Разделение данных
            random.shuffle(image_files)
            split_idx = int(len(image_files) * 0.8)
            train_images = image_files[:split_idx]
            val_images = image_files[split_idx:]
            
            # Валидация и перемещение файлов
            for img_path in train_images + val_images:
                label_path = img_path.with_suffix('.txt')
                if not label_path.exists():
                    self.logger.warning(f"Отсутствует файл разметки для {img_path}")
                    continue
                
                target_dir = train_dir if img_path in train_images else val_dir
                try:
                    shutil.move(str(img_path), str(target_dir / img_path.name))
                    shutil.move(str(label_path), str(target_dir / label_path.name))
                except Exception as e:
                    self.logger.error(f"Ошибка при перемещении файлов: {str(e)}")
                    continue
            
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}\n{traceback.format_exc()}")
            return False

    async def train_model(self) -> bool:
        """Обучение модели с расширенными параметрами"""
        try:
            if not self._prepare_training_data():
                return False
            
            # Создание директории для результатов
            run_dir = OUTPUT_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.fs_manager.ensure_directories(run_dir)
            
            # Конфигурация обучения
            yaml_content = {
                'path': str(DATASET_DIR),
                'train': 'train',
                'val': 'val',
                'names': {
                    0: 'bridge_closed',
                    1: 'bridge_open'
                }
            }
            
            # Сохранение конфигурации
            data_yaml_path = run_dir / "data.yaml"
            with open(data_yaml_path, 'w') as f:
                yaml.dump(yaml_content, f)
            
            # Запуск обучения с расширенными параметрами
            self.logger.info("Запуск процесса обучения...")
            self.model.train(
                data=str(data_yaml_path),
                epochs=100,
                imgsz=640,
                batch=16,
                patience=10,
                save=True,
                project=str(run_dir),
                name="train",
                exist_ok=True,
                pretrained=True,
                optimizer='Adam',
                lr0=0.001,
                weight_decay=0.0005,
                warmup_epochs=3,
                cos_lr=True,
                workers=4,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Загрузка новой модели
            best_model_path = run_dir / "train" / "weights" / "best.pt"
            if best_model_path.exists():
                self.model = self.model_manager.load_model(best_model_path)
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}\n{traceback.format_exc()}")
            return False

class BridgeBot:
    """Улучшенная версия телеграм-бота"""
    def __init__(self):
        self.detector = BridgeDetectorV3()
        self.logger = logging.getLogger(__name__)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        keyboard = [
            [KeyboardButton("Проверить статус")],
            [KeyboardButton("Помощь")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        await update.message.reply_text(
            "Привет! Я бот для определения статуса моста.\n"
            "Используйте кнопки или команды:\n"
            "/status - проверить текущий статус\n"
            "/help - получить справку\n"
            "/train - начать режим обучения\n",
            reply_markup=reply_markup
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды помощи"""
        await update.message.reply_text(
            "Доступные команды:\n"
            "/status - проверить текущий статус моста\n"
            "/train - начать режим обучения\n"
            "/stop_training - закончить режим обучения\n"
            "/help - показать это сообщение"
        )

    async def check_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Проверка текущего статуса моста"""
        try:
            frame = await self.detector.capture_frame()
            if frame is None:
                await update.message.reply_text("Не удалось получить кадр с камеры")
                return

            # Сохраняем кадр во временную директорию
            temp_path = TEMP_DIR / f"status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(temp_path), frame)

            # Делаем предсказание
            results = self.detector.model.predict(
                source=str(temp_path),
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD
            )

            # Получаем результаты
            result = results[0]
            if len(result.boxes) > 0:
                # Получаем класс с наибольшей уверенностью
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                max_conf_idx = confidences.argmax()
                
                class_id = int(classes[max_conf_idx])
                confidence = confidences[max_conf_idx]
                
                status = "открыт" if class_id == 1 else "закрыт"
                await update.message.reply_text(
                    f"Статус моста: {status}\n"
                    f"Уверенность: {confidence:.2%}"
                )
            else:
                await update.message.reply_text(
                    "Не удалось определить статус моста на изображении"
                )

            # Отправляем изображение
            await update.message.reply_photo(temp_path.open('rb'))

        except Exception as e:
            self.logger.error(f"Ошибка при проверке статуса: {str(e)}")
            await update.message.reply_text(f"Произошла ошибка: {str(e)}")
        finally:
            # Удаляем временный файл
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()

    async def start_training(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Начало режима обучения"""
        self.detector.is_training_mode = True
        await update.message.reply_text(
            "Режим обучения активирован.\n"
            "Теперь я буду сохранять все кадры для обучения.\n"
            "Используйте /stop_training для завершения."
        )

    async def stop_training(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Завершение режима обучения"""
        if not self.detector.is_training_mode:
            await update.message.reply_text("Режим обучения не был активирован.")
            return

        try:
            self.detector.is_training_mode = False
            await update.message.reply_text(
                "Режим обучения завершен.\n"
                "Начинаю процесс обучения модели..."
            )
            
            # Запускаем обучение в фоновом режиме
            asyncio.create_task(self._train_model(update))
            
        except Exception as e:
            self.logger.error(f"Ошибка при остановке режима обучения: {str(e)}")
            await update.message.reply_text(f"Произошла ошибка: {str(e)}")

    async def _train_model(self, update: Update):
        """Фоновая задача для обучения модели"""
        try:
            # Подготовка данных
            self.detector._prepare_training_data()
            
            # Создаем директорию для результатов
            run_dir = OUTPUT_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Создаем data.yaml
            yaml_content = {
                'path': str(DATASET_DIR),
                'train': 'train',
                'val': 'val',
                'names': {
                    0: 'bridge_closed',
                    1: 'bridge_open'
                }
            }
            
            data_yaml_path = run_dir / "data.yaml"
            with open(data_yaml_path, 'w') as f:
                yaml.dump(yaml_content, f)
            
            # Запускаем обучение
            self.detector.model.train(
                data=str(data_yaml_path),
                epochs=100,
                imgsz=640,
                batch=16,
                patience=10,
                save=True,
                project=str(run_dir),
                name="train",
                exist_ok=True
            )
            
            # Копируем лучшую модель
            best_model = run_dir / "train" / "weights" / "best.pt"
            if best_model.exists():
                shutil.copy(best_model, MODELS_DIR / "best.pt")
                await update.message.reply_text("Обучение завершено успешно! Модель обновлена.")
            else:
                await update.message.reply_text("Обучение завершено, но лучшая модель не найдена.")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            await update.message.reply_text(f"Ошибка при обучении модели: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений"""
        text = update.message.text.lower()
        if text == "проверить статус":
            await self.check_status(update, context)
        elif text == "помощь":
            await self.help(update, context)
        else:
            await update.message.reply_text(
                "Извините, я не понимаю эту команду.\n"
                "Используйте кнопки или /help для списка команд."
            )

def retry_on_exception(retries: int = 3, delay: float = 1.0):
    """Декоратор для повторных попыток выполнения функции при ошибках"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Попытка {attempt + 1}/{retries} не удалась: {str(e)}")
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            logger.error(f"Все попытки исчерпаны. Последняя ошибка: {str(last_exception)}")
            raise last_exception
        return wrapper
    return decorator

def setup_commands(application: Application):
    """Настройка команд бота"""
    bot = BridgeBot()
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help))
    application.add_handler(CommandHandler("check", bot.check_status))
    application.add_handler(CommandHandler("train_start", bot.start_training))
    application.add_handler(CommandHandler("train_stop", bot.stop_training))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

async def main():
    """Основная функция"""
    try:
        # Настройка логирования с относительным путем
        FileSystemManager.setup_logging(
            Path('logs/bot.log'),
            max_size_mb=10,
            backup_count=5
        )
        
        # Создание и настройка приложения
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        setup_commands(application)
        
        # Запуск бота
        logger.info("Бот запущен")
        await application.run_polling()
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
