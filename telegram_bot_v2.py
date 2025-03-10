import os
import cv2
import json
import numpy as np
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError
from ultralytics import YOLO
from loguru import logger
import time
from pathlib import Path
import shutil
import random
import logging
import torch
import yaml
from config import (
    TELEGRAM_TOKEN,
    CAMERA_URL,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    BASE_DIR,
    DATASET_DIR,
    MODELS_DIR,
    TEMP_DIR
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Отключаем логи от httpx
logging.getLogger('httpx').setLevel(logging.WARNING)

# Создание необходимых директорий
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class BridgeDetectorV2:
    def __init__(self):
        """Инициализация детектора мостов версии 2.0"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация BridgeDetectorV2")
        self.model = self._load_model()
        self.camera_url = CAMERA_URL
        self.training_data = {}  # Словарь для хранения данных обучения
        self.is_training_mode = False  # Флаг режима обучения

    def _load_model(self):
        """Загрузка модели YOLO"""
        model_path = MODELS_DIR / "best.pt"
        if model_path.exists():
            self.logger.info(f"Загрузка существующей модели: {model_path}")
            return YOLO(str(model_path))
        else:
            self.logger.info("Создание новой модели YOLOv8n")
            return YOLO('yolov8n.pt')

    def _save_frame(self, frame, label):
        """Сохранение кадра для обучения"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = TEMP_DIR / f"bridge_{timestamp}.jpg"
        label_path = TEMP_DIR / f"bridge_{timestamp}.txt"
        
        # Сохраняем изображение
        cv2.imwrite(str(image_path), frame)
        
        # Получаем размеры изображения
        height, width = frame.shape[:2]
        
        # Создаем аннотацию в формате YOLO
        # class_id x_center y_center width height
        # Для простоты берем центр изображения и фиксированный размер
        class_id = 0 if label == 'closed' else 1  # 0 - закрыт, 1 - открыт
        x_center = 0.5  # центр по X
        y_center = 0.5  # центр по Y
        box_width = 0.8  # ширина бокса
        box_height = 0.8  # высота бокса
        
        # Записываем аннотацию
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}")
        
        return image_path

    def start_training_mode(self):
        """Включение режима обучения"""
        self.is_training_mode = True
        self.training_data = {}
        self.logger.info("Режим обучения активирован")

    def _cleanup_old_training_dirs(self):
        """Очистка старых директорий обучения, оставляя только последнюю"""
        training_dirs = sorted([d for d in MODELS_DIR.glob('bridge_detector*') if d.is_dir()])
        if len(training_dirs) > 1:  # Оставляем только последнюю директорию
            for old_dir in training_dirs[:-1]:
                try:
                    shutil.rmtree(old_dir)
                    self.logger.info(f"Удалена старая директория обучения: {old_dir}")
                except Exception as e:
                    self.logger.error(f"Ошибка при удалении директории {old_dir}: {e}")

    def stop_training_mode(self):
        """Остановка режима обучения"""
        try:
            self.logger.info("Завершение режима обучения...")
            
            # Определяем следующее имя директории для результатов
            existing_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("bridge_detector")]
            if existing_dirs:
                # Извлекаем номера из имен директорий и находим максимальный
                dir_numbers = []
                for d in existing_dirs:
                    try:
                        num = int(d.name.split("bridge_detector")[1])
                        dir_numbers.append(num)
                    except (ValueError, IndexError):
                        continue
                
                if dir_numbers:
                    next_num = max(dir_numbers) + 1
                else:
                    next_num = 1
            else:
                next_num = 1
            
            run_name = f"bridge_detector{next_num}"
            self.logger.info(f"Создание директории для результатов: {run_name}")
            run_dir = MODELS_DIR / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Копируем текущее изображение, если оно существует
            img_path = TEMP_DIR / "current_status.jpg"
            if img_path.exists():
                self.logger.info(f"Копирование текущего изображения: {img_path}")
                shutil.copy2(img_path, run_dir / "current_status.jpg")
            else:
                self.logger.warning(f"Файл {img_path} не найден, пропускаем копирование")
            
            # Копируем файлы датасета
            self.logger.info("Копирование файлов датасета...")
            dataset_dir = run_dir / "dataset"
            dataset_dir.mkdir(exist_ok=True)
            
            # Копируем train и val директории
            for split in ["train", "val"]:
                split_dir = dataset_dir / split
                split_dir.mkdir(exist_ok=True)
                source_dir = DATASET_DIR / split
                if source_dir.exists():
                    self.logger.info(f"Копирование {split} данных...")
                    for file in source_dir.glob("*"):
                        if file.is_file():
                            shutil.copy2(file, split_dir)
                else:
                    self.logger.warning(f"Директория {source_dir} не найдена")
            
            # Создаем data.yaml файл
            yaml_content = {
                'path': str(dataset_dir),
                'train': 'train',
                'val': 'val',
                'names': {
                    0: 'bridge_closed',
                    1: 'bridge_open'
                }
            }
            with open(dataset_dir / "data.yaml", 'w') as f:
                yaml.dump(yaml_content, f)
            
            # Запускаем обучение
            self.logger.info("Запуск процесса обучения...")
            self.model.train(
                data=str(dataset_dir / "data.yaml"),
                epochs=100,
                imgsz=640,
                batch=16,
                patience=10,
                save=True,
                project=str(run_dir),
                name="weights",
                exist_ok=True
            )
            
            # Загружаем новую модель
            self.logger.info("Загрузка обученной модели...")
            new_model_path = run_dir / "weights" / "best.pt"
            if not new_model_path.exists():
                self.logger.error(f"Модель не найдена по пути: {new_model_path}")
                return False
                
            self.model = YOLO(str(new_model_path))
            self.logger.info("Модель успешно загружена")
            
            # Копируем лучшую модель в корневую директорию
            self.logger.info("Копирование лучшей модели...")
            shutil.copy2(new_model_path, MODELS_DIR / "best.pt")
            
            # Очищаем старые директории обучения
            self._cleanup_old_training_dirs()
            
            self.logger.info("Режим обучения успешно завершен")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при завершении режима обучения: {str(e)}", exc_info=True)
            return False

    def get_current_frame(self):
        """Получение текущего кадра с камеры"""
        try:
            cap = cv2.VideoCapture(self.camera_url)
            if not cap.isOpened():
                raise ValueError("Не удалось подключиться к камере")

            ret, frame = cap.read()
            if not ret:
                raise ValueError("Не удалось получить кадр")

            cap.release()
            return frame
        except Exception as e:
            self.logger.error(f"Ошибка при получении кадра: {str(e)}")
            return None

    def process_frame(self, frame, save_path=None):
        """Обработка кадра и определение состояния моста"""
        try:
            # Получаем предсказания
            results = self.model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )[0]
            
            # Получаем детекции
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = 'bridge_open' if class_id == 1 else 'bridge_closed'
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
            
            # Логируем детекции для отладки
            self.logger.info(f"Найдено детекций: {len(detections)}")
            for det in detections:
                self.logger.info(f"Детекция: {det['class']} с уверенностью {det['confidence']:.2f}")
            
            # Определяем статус
            if not detections:
                self.logger.warning("Детекций не найдено, статус: unknown")
                status = 'unknown'
            else:
                # Берем детекцию с наибольшей уверенностью
                best_detection = max(detections, key=lambda x: x['confidence'])
                status = 'open' if best_detection['class'] == 'bridge_open' else 'closed'
                self.logger.info(f"Определен статус: {status} (уверенность: {best_detection['confidence']:.2f})")
            
            # Отрисовываем результаты
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                color = (0, 255, 0) if det['class'] == 'bridge_open' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Добавляем общий статус
            status_text = f"Bridge Status: {status.upper()}"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Сохраняем результат если указан путь
            if save_path:
                cv2.imwrite(save_path, frame)
            
            return status, frame
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке кадра: {str(e)}")
            return 'unknown', frame

class BridgeBot:
    def __init__(self):
        """Инициализация бота"""
        self.detector = BridgeDetectorV2()
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        keyboard = [
            [KeyboardButton("Проверить статус")],
            [KeyboardButton("Начать обучение"), KeyboardButton("Завершить обучение")],
            [KeyboardButton("Помощь")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            'Привет! Я бот для мониторинга статуса моста v2.0\n'
            'Теперь я умею учиться! Используйте кнопки ниже для навигации:',
            reply_markup=reply_markup
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды помощи"""
        help_text = (
            "🤖 Бот для мониторинга статуса моста v2.0\n\n"
            "Основные команды:\n"
            "🔍 Проверить статус - получить текущее состояние моста\n"
            "📚 Начать обучение - войти в режим обучения\n"
            "🎓 Завершить обучение - закончить обучение и обновить модель\n\n"
            "В режиме обучения:\n"
            "1. Бот будет присылать кадры с камеры\n"
            "2. Используйте кнопки Открыт/Закрыт для разметки\n"
            "3. После накопления достаточного количества данных,\n"
            "   нажмите 'Завершить обучение'"
        )
        await update.message.reply_text(help_text)

    async def check_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Проверка текущего статуса моста"""
        try:
            await update.message.reply_text("Получаю текущий статус моста...")
            
            # Получаем кадр
            frame = self.detector.get_current_frame()
            if frame is None:
                await update.message.reply_text("Не удалось получить изображение с камеры")
                return
            
            # Сохраняем результат
            output_path = str(TEMP_DIR / "current_status.jpg")
            status, processed_frame = self.detector.process_frame(frame, output_path)
            
            status_text = {
                'open': 'Мост открыт 🟢',
                'closed': 'Мост закрыт 🔴',
                'unknown': 'Статус неизвестен ⚠️'
            }[status]

            # Создаем inline кнопки для валидации
            keyboard = [
                [
                    InlineKeyboardButton("✅ Верно", callback_data=f"validate_correct_{status}"),
                    InlineKeyboardButton("❌ Неверно", callback_data="validate_incorrect")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            with open(output_path, 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"Текущий статус моста: {status_text}\n\nПожалуйста, подтвердите правильность определения:",
                    reply_markup=reply_markup
                )
        except Exception as e:
            logger.error(f"Ошибка при проверке статуса: {str(e)}")
            await update.message.reply_text("Произошла ошибка при проверке статуса")

    async def start_training(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Начало режима обучения"""
        self.detector.start_training_mode()
        await update.message.reply_text(
            "Режим обучения активирован! Я буду присылать кадры для разметки.\n"
            "Используйте кнопки Открыт/Закрыт для каждого кадра."
        )
        await self.send_frame_for_training(update.effective_message, context)

    async def send_frame_for_training(self, message, context: ContextTypes.DEFAULT_TYPE):
        """Отправка кадра для разметки"""
        frame = self.detector.get_current_frame()
        if frame is None:
            await context.bot.send_message(
                chat_id=message.chat.id,
                text="Не удалось получить кадр с камеры"
            )
            return
        
        # Сохраняем временный файл
        temp_path = str(TEMP_DIR / "temp_frame.jpg")
        cv2.imwrite(temp_path, frame)
        
        # Создаем inline кнопки
        keyboard = [
            [
                InlineKeyboardButton("Открыт", callback_data="label_open"),
                InlineKeyboardButton("Закрыт", callback_data="label_closed")
            ],
            [InlineKeyboardButton("Пропустить", callback_data="skip")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        with open(temp_path, 'rb') as photo:
            await context.bot.send_photo(
                chat_id=message.chat.id,
                photo=photo,
                caption="Укажите состояние моста на кадре:",
                reply_markup=reply_markup
            )

    async def handle_training_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка ответов в режиме обучения"""
        query = update.callback_query
        await query.answer()
        
        # Проверяем, что это callback для режима обучения
        if not query.data.startswith(("label_", "skip")):
            return
        
        if not self.detector.is_training_mode:
            await query.edit_message_caption(
                caption="Режим обучения не активен"
            )
            return
        
        if query.data.startswith("label_"):
            label = query.data.replace("label_", "")
            
            # Сохраняем кадр с меткой
            frame = cv2.imread(str(TEMP_DIR / "temp_frame.jpg"))
            if frame is not None:
                self.detector._save_frame(frame, label)
                await query.edit_message_caption(
                    caption=f"Спасибо! Кадр сохранен с меткой: {label}"
                )
                
                # Отправляем следующий кадр
                await self.send_frame_for_training(query.message, context)
        elif query.data == "skip":
            await query.edit_message_caption(
                caption="Кадр пропущен"
            )
            await self.send_frame_for_training(query.message, context)

    async def stop_training(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Завершение режима обучения"""
        if not self.detector.is_training_mode:
            await update.message.reply_text("Режим обучения не был активирован")
            return
        
        await update.message.reply_text("Завершаю режим обучения и начинаю обучение модели...")
        
        success = self.detector.stop_training_mode()
        if success:
            await update.message.reply_text(
                "Обучение успешно завершено! Модель обновлена и готова к использованию."
            )
        else:
            await update.message.reply_text(
                "Произошла ошибка при обучении модели. Попробуйте позже."
            )

    async def handle_validation_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка валидации предсказаний"""
        query = update.callback_query
        await query.answer()
        
        if query.data.startswith("validate_correct_"):
            # Если предсказание верное, сохраняем кадр с текущей меткой
            status = query.data.replace("validate_correct_", "")
            frame = cv2.imread(str(TEMP_DIR / "current_status.jpg"))
            if frame is not None:
                self.detector._save_frame(frame, status)
                await query.edit_message_caption(
                    caption=f"Спасибо за подтверждение! Кадр сохранен для обучения с меткой: {status}"
                )
        
        elif query.data == "validate_incorrect":
            # Если предсказание неверное, спрашиваем правильный статус
            keyboard = [
                [
                    InlineKeyboardButton("Мост открыт", callback_data="correct_status_open"),
                    InlineKeyboardButton("Мост закрыт", callback_data="correct_status_closed")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_caption(
                caption="Пожалуйста, укажите правильный статус моста:",
                reply_markup=reply_markup
            )
        
        elif query.data.startswith("correct_status_"):
            # Сохраняем кадр с исправленной меткой
            correct_status = query.data.replace("correct_status_", "")
            frame = cv2.imread(str(TEMP_DIR / "current_status.jpg"))
            if frame is not None:
                self.detector._save_frame(frame, correct_status)
                await query.edit_message_caption(
                    caption=f"Спасибо за исправление! Кадр сохранен для обучения с меткой: {correct_status}"
                )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений"""
        text = update.message.text
        
        if text == "Проверить статус":
            await self.check_status(update, context)
        elif text == "Начать обучение":
            await self.start_training(update, context)
        elif text == "Завершить обучение":
            await self.stop_training(update, context)
        elif text == "Помощь":
            await self.help(update, context)
        else:
            await update.message.reply_text(
                "Используйте кнопки для навигации"
            )

async def setup_commands(bot):
    """Установка команд бота"""
    commands = [
        ("start", "Запустить бота и показать основное меню"),
        ("status", "Проверить текущий статус моста"),
        ("train", "Начать режим обучения"),
        ("stop_train", "Завершить обучение и обновить модель"),
        ("help", "Показать справку по командам")
    ]
    try:
        await bot.set_my_commands(commands)
        logger.info("Команды бота установлены")
    except Exception as e:
        logger.error(f"Ошибка при установке команд: {e}")

def main():
    """Основная функция"""
    bot = BridgeBot()
    
    # Создаем приложение
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .build()
    )

    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("status", bot.check_status))
    application.add_handler(CommandHandler("train", bot.start_training))
    application.add_handler(CommandHandler("stop_train", bot.stop_training))
    application.add_handler(CommandHandler("help", bot.help))
    
    # Добавляем обработчики сообщений и callback'ов
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    
    # Обработчики callback'ов с правильным порядком и фильтрацией
    application.add_handler(CallbackQueryHandler(bot.handle_validation_callback, pattern="^validate_"))
    application.add_handler(CallbackQueryHandler(bot.handle_validation_callback, pattern="^correct_status_"))
    application.add_handler(CallbackQueryHandler(bot.handle_training_callback))  # Этот обработчик последний

    # Запускаем бота
    logger.info("Запуск бота v2.0...")
    
    # Устанавливаем команды при старте
    async def post_init(application: Application) -> None:
        await setup_commands(application.bot)
    
    # Запускаем бота
    application.post_init = post_init
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
