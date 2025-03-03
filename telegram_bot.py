import os
import cv2
import numpy as np
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError
from ultralytics import YOLO
from loguru import logger
import time
from config import *

class BridgeDetector:
    def __init__(self):
        """Инициализация детектора мостов"""
        logger.info("Инициализация BridgeDetector")
        self._ensure_directories()
        self.model = self._load_model()
        self.camera_url = "https://node007.youlook.ru/cam001544/index.m3u8?token=f2d6d079b8fc54a745e085b0c34e2e08"

    def _ensure_directories(self):
        """Создание необходимых директорий"""
        for directory in [OUTPUT_DIR, MODELS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Создана директория: {directory}")

    def _load_model(self):
        """Загрузка модели YOLO"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
        logger.info(f"Загрузка модели из: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        
        # Проверяем информацию о модели
        logger.info(f"Модель загружена. Доступные классы модели: {model.names}")
        logger.info(f"Доступные классы в конфиге: {CLASSES}")
        logger.info(f"Порог уверенности: {CONFIDENCE_THRESHOLD}")
        logger.info(f"Порог IoU: {IOU_THRESHOLD}")
        
        return model

    def _determine_bridge_status(self, detections, image_shape):
        """Определение статуса моста на основе детекций"""
        if not detections:
            logger.warning("Нет детекций на изображении")
            return 'unknown'
            
        # Фильтруем детекции по уверенности
        bridge_detections = []
        for det in detections:
            if det['confidence'] > CONFIDENCE_THRESHOLD:
                bridge_detections.append(det)
        
        if not bridge_detections:
            logger.warning("Нет детекций с достаточной уверенностью")
            return 'unknown'
            
        # Сортируем детекции по уверенности
        bridge_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Берем самую уверенную детекцию
        best_detection = bridge_detections[0]
        
        # Анализ геометрических характеристик
        x1, y1, x2, y2 = best_detection['bbox']
        width = x2 - x1
        height = y2 - y1
        area = width * height
        image_area = image_shape[0] * image_shape[1]
        area_ratio = area / image_area
        
        # Вычисляем центр по вертикали
        center_y = (y1 + y2) / 2
        vertical_position = center_y / image_shape[0]
        
        # Вычисляем соотношение сторон
        aspect_ratio = width / height if height > 0 else 0
        
        logger.info(f"Анализ детекции: confidence={best_detection['confidence']:.2f}, "
                   f"area_ratio={area_ratio:.2f}, vertical_position={vertical_position:.2f}, "
                   f"aspect_ratio={aspect_ratio:.2f}")
        
        # Определение состояния моста
        if vertical_position < 0.35:  # Мост в верхней трети экрана
            logger.info("Мост определен как открытый (высокое положение)")
            return 'open'
        elif vertical_position > 0.65:  # Мост в нижней трети экрана
            logger.info("Мост определен как закрытый (низкое положение)")
            return 'closed'
        elif area_ratio > 0.4:  # Большая площадь детекции
            if vertical_position > 0.5:  # В нижней половине
                logger.info("Мост определен как закрытый (большая площадь в нижней части)")
                return 'closed'
        elif area_ratio < 0.3:  # Малая площадь детекции
            if vertical_position < 0.5:  # В верхней половине
                if aspect_ratio > 1.2:  # Горизонтально вытянутый
                    logger.info("Мост определен как открытый (малая площадь в верхней части)")
                    return 'open'
        
        logger.warning(f"Неоднозначное состояние моста: "
                      f"area_ratio={area_ratio:.2f}, vertical_position={vertical_position:.2f}")
        return 'unknown'

    def get_current_status(self):
        """Получение текущего статуса моста"""
        try:
            # Подключаемся к камере
            cap = cv2.VideoCapture(self.camera_url)
            if not cap.isOpened():
                raise ValueError("Не удалось подключиться к камере")

            # Получаем кадр
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Не удалось получить кадр")

            # Проверяем размер изображения
            height, width = frame.shape[:2]
            logger.info(f"Размер изображения: {width}x{height}")

            # Сохраняем исходное изображение для отладки
            debug_path = os.path.join(OUTPUT_DIR, "debug_input.jpg")
            cv2.imwrite(debug_path, frame)
            logger.info(f"Сохранено отладочное изображение: {debug_path}")

            # Получаем предсказания
            results = self.model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )[0]
            
            # Преобразуем результаты в удобный формат
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"unknown_class_{class_id}"
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })

            logger.info(f"Найдено детекций: {len(detections)}")
            for det in detections:
                logger.info(f"Детекция: класс={det['class']}, уверенность={det['confidence']:.2f}")

            # Определяем статус моста
            status = self._determine_bridge_status(detections, frame.shape[:2])
            
            # Отрисовываем детекции
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Добавляем текст с классом и уверенностью
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Добавляем текст со статусом
            status_text = f"Bridge Status: {status.upper()}"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Сохраняем кадр
            output_path = os.path.join(OUTPUT_DIR, "current_status.jpg")
            cv2.imwrite(output_path, frame)
            
            # Освобождаем ресурсы
            cap.release()
            
            return status, output_path
            
        except Exception as e:
            logger.error(f"Ошибка при получении статуса: {str(e)}")
            return None, None

# Инициализация детектора
detector = BridgeDetector()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    try:
        keyboard = [
            [KeyboardButton("Проверить статус моста")],
            [KeyboardButton("Помощь")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            'Привет! Я бот для мониторинга статуса моста.\n'
            'Используйте кнопки ниже для навигации:',
            reply_markup=reply_markup
        )
    except (TimedOut, NetworkError) as e:
        logger.error(f"Ошибка сети при отправке сообщения: {e}")
        await update.message.reply_text(
            "Произошла ошибка сети. Пожалуйста, попробуйте позже."
        )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик запроса статуса моста"""
    try:
        await update.message.reply_text("Получаю текущий статус моста...")
        
        status, image_path = detector.get_current_status()
        if status and image_path:
            status_text = {
                'open': 'Мост открыт 🟢',
                'closed': 'Мост закрыт 🔴',
                'unknown': 'Статус моста неизвестен ⚠️'
            }.get(status, 'Статус моста неизвестен ⚠️')
            
            with open(image_path, 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"Текущий статус моста: {status_text}"
                )
        else:
            await update.message.reply_text("Не удалось получить статус моста. Попробуйте позже.")
    except (TimedOut, NetworkError) as e:
        logger.error(f"Ошибка сети при отправке статуса: {e}")
        await update.message.reply_text(
            "Произошла ошибка сети. Пожалуйста, попробуйте позже."
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        await update.message.reply_text(
            "Произошла ошибка. Пожалуйста, попробуйте позже."
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик запроса помощи"""
    try:
        help_text = (
            "🤖 Бот для мониторинга статуса моста\n\n"
            "Статусы моста:\n"
            "🟢 - Мост открыт\n"
            "🔴 - Мост закрыт\n"
            "⚠️ - Статус неизвестен\n\n"
            "Используйте кнопку 'Проверить статус моста' для получения актуальной информации."
        )
        await update.message.reply_text(help_text)
    except (TimedOut, NetworkError) as e:
        logger.error(f"Ошибка сети при отправке справки: {e}")
        await update.message.reply_text(
            "Произошла ошибка сети. Пожалуйста, попробуйте позже."
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    try:
        text = update.message.text
        
        if text == "Проверить статус моста":
            await status(update, context)
        elif text == "Помощь":
            await help_command(update, context)
        else:
            await update.message.reply_text(
                "Используйте кнопки ниже для навигации",
                reply_markup=ReplyKeyboardMarkup(
                    [["Проверить статус моста"], ["Помощь"]],
                    resize_keyboard=True
                )
            )
    except (TimedOut, NetworkError) as e:
        logger.error(f"Ошибка сети при обработке сообщения: {e}")
        await update.message.reply_text(
            "Произошла ошибка сети. Пожалуйста, попробуйте позже."
        )

def main():
    """Основная функция"""
    # Создаем приложение с увеличенными таймаутами
    application = (
        Application.builder()
        .token("6336113851:AAGJqgNAQKYwCldn4e4vE3y7AC_FYm9taI4")
        .connect_timeout(30.0)  # Увеличиваем таймаут подключения
        .read_timeout(30.0)     # Увеличиваем таймаут чтения
        .write_timeout(30.0)    # Увеличиваем таймаут записи
        .build()
    )

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота
    logger.info("Запуск бота...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main() 