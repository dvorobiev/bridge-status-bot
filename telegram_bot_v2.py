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

# Конфигурация
TELEGRAM_TOKEN = "6336113851:AAGJqgNAQKYwCldn4e4vE3y7AC_FYm9taI4"
CAMERA_URL = "https://node007.youlook.ru/cam001544/index.m3u8?token=f2d6d079b8fc54a745e085b0c34e2e08"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.3

# Пути к директориям
BASE_DIR = Path("bridge_detector_v2")
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Создаем все необходимые директории
for dir_path in [MODELS_DIR, TRAIN_DIR, VAL_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class BridgeDetectorV2:
    def __init__(self):
        """Инициализация детектора мостов версии 2.0"""
        logger.info("Инициализация BridgeDetectorV2")
        self.model = self._load_model()
        self.camera_url = CAMERA_URL
        self.training_data = {}  # Словарь для хранения данных обучения
        self.is_training_mode = False  # Флаг режима обучения

    def _load_model(self):
        """Загрузка модели YOLO"""
        model_path = MODELS_DIR / "best.pt"
        if model_path.exists():
            logger.info(f"Загрузка существующей модели: {model_path}")
            return YOLO(str(model_path))
        else:
            logger.info("Создание новой модели YOLOv8n")
            return YOLO('yolov8n.pt')

    def _save_frame(self, frame, label):
        """Сохранение кадра для обучения"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = TRAIN_DIR / f"bridge_{timestamp}.jpg"
        label_path = TRAIN_DIR / f"bridge_{timestamp}.txt"
        
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
        logger.info("Режим обучения активирован")

    def stop_training_mode(self):
        """Выключение режима обучения и запуск обучения модели"""
        self.is_training_mode = False
        logger.info("Режим обучения деактивирован")
        
        # Получаем список всех файлов в train директории
        train_files = list(TRAIN_DIR.glob('*.jpg'))
        if not train_files:
            logger.error("Нет данных для обучения")
            return False
            
        # Очищаем директорию валидации
        for file in VAL_DIR.glob('*'):
            file.unlink()
            
        # Выбираем 20% файлов для валидации
        val_count = max(1, int(len(train_files) * 0.2))
        val_files = random.sample(train_files, val_count)
        
        # Копируем выбранные файлы и их аннотации в val директорию
        for img_path in val_files:
            # Копируем изображение
            shutil.copy2(img_path, VAL_DIR)
            # Копируем соответствующий txt файл
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                shutil.copy2(txt_path, VAL_DIR)
        
        # Получаем абсолютный путь к директории датасета
        dataset_dir = DATASET_DIR.resolve()
        
        # Создаем yaml файл для обучения
        dataset_yaml = DATASET_DIR / "bridge.yaml"
        yaml_content = {
            'path': str(dataset_dir),  # Абсолютный путь к корневой директории датасета
            'train': 'train',  # Относительный путь к train от path
            'val': 'val',      # Относительный путь к val от path
            'names': {
                0: 'bridge_closed',
                1: 'bridge_open'
            }
        }
        
        with open(dataset_yaml, 'w') as f:
            import yaml
            yaml.dump(yaml_content, f)
        
        logger.info(f"Датасет подготовлен: {len(train_files)} файлов для обучения, {len(val_files)} для валидации")
        
        # Запускаем обучение
        logger.info("Начало обучения модели...")
        try:
            results = self.model.train(
                data=str(dataset_yaml),
                epochs=50,
                imgsz=640,
                batch=16,
                patience=10,
                project=str(MODELS_DIR),
                name="bridge_detector"
            )
            logger.info("Обучение завершено успешно")
            
            # Загружаем новую модель
            new_model_path = MODELS_DIR / "bridge_detector" / "weights" / "best.pt"
            if new_model_path.exists():
                logger.info(f"Загружаем новую модель из {new_model_path}")
                self.model = YOLO(str(new_model_path))
                # Копируем лучшую модель в корень для следующих запусков
                shutil.copy2(new_model_path, MODELS_DIR / "best.pt")
                logger.info("Новая модель загружена и сохранена как best.pt")
            else:
                logger.error(f"Файл модели не найден по пути {new_model_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Ошибка при обучении: {str(e)}")
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
            logger.error(f"Ошибка при получении кадра: {str(e)}")
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
            logger.info(f"Найдено детекций: {len(detections)}")
            for det in detections:
                logger.info(f"Детекция: {det['class']} с уверенностью {det['confidence']:.2f}")
            
            # Определяем статус
            if not detections:
                logger.warning("Детекций не найдено, статус: unknown")
                status = 'unknown'
            else:
                # Берем детекцию с наибольшей уверенностью
                best_detection = max(detections, key=lambda x: x['confidence'])
                status = 'open' if best_detection['class'] == 'bridge_open' else 'closed'
                logger.info(f"Определен статус: {status} (уверенность: {best_detection['confidence']:.2f})")
            
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
            logger.error(f"Ошибка при обработке кадра: {str(e)}")
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
            output_path = str(OUTPUT_DIR / "current_status.jpg")
            status, processed_frame = self.detector.process_frame(frame, output_path)
            
            status_text = {
                'open': 'Мост открыт 🟢',
                'closed': 'Мост закрыт 🔴',
                'unknown': 'Статус неизвестен ⚠️'
            }[status]

            # Сохраняем оригинальный кадр для возможного обучения
            temp_path = str(TEMP_DIR / "status_frame.jpg")
            cv2.imwrite(temp_path, frame)
            
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
            frame = cv2.imread(str(TEMP_DIR / "status_frame.jpg"))
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
            frame = cv2.imread(str(TEMP_DIR / "status_frame.jpg"))
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
