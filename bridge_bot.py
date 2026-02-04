#!/usr/bin/env python3
"""
Telegram-бот для мониторинга понтонного моста.
Использует Gemini Vision для определения статуса: СВЕДЁН или РАЗВЕДЁН.
"""

import os
import io
import asyncio
import logging
import subprocess
import tempfile
from datetime import datetime
from dotenv import load_dotenv
import httpx
import re

# Import for backward compatibility - future versions should use google.genai
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Загружаем переменные окружения
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CAMERA_URL = os.getenv("CAMERA_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 5))  # минуты

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Настройка Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Глобальное состояние
last_status = None
subscribers = set()  # chat_id подписчиков на уведомления


# Настройки камеры rtsp.me (Митяевский мост)
RTSP_ME_EMBED_URL = "https://rtsp.me/embed/yEDF9iDT/"


def get_rtspme_stream_url() -> str | None:
    """Получает актуальный m3u8 URL из rtsp.me embed страницы."""
    try:
        with httpx.Client(timeout=15, verify=False) as client:
            response = client.get(RTSP_ME_EMBED_URL, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code == 200:
                # Ищем m3u8 URL в HTML
                match = re.search(r'https://msk\.rtsp\.me/[^"\']+\.m3u8[^"\']*', response.text)
                if match:
                    return match.group(0)
    except Exception as e:
        logger.error(f"Failed to get rtsp.me stream URL: {e}")
    return None


def capture_frame() -> bytes | None:
    """Захватывает кадр с камеры rtsp.me через ffmpeg."""
    try:
        # Получаем актуальный URL потока
        stream_url = get_rtspme_stream_url()
        if not stream_url:
            logger.error("Failed to get stream URL from rtsp.me")
            return None

        logger.info(f"Capturing frame from: {stream_url[:60]}...")

        # Захватываем кадр через ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-i", stream_url,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "2",
            "pipe:1"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode == 0 and result.stdout:
            logger.info(f"Frame captured: {len(result.stdout)} bytes")
            return result.stdout

        logger.error(f"ffmpeg error: {result.stderr.decode()[:200]}")
        return None

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout")
        return None
    except Exception as e:
        logger.error(f"Capture error: {e}")
        return None


def analyze_bridge(image_bytes: bytes) -> dict:
    """Отправляет изображение в Gemini и получает статус моста и светофора."""
    try:
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }

        prompt = """Посмотри на изображение с камеры понтонного моста.

Определи:
1. Статус моста: СВЕДЁН (можно проехать) или РАЗВЕДЁН (есть разрыв)
2. Цвет светофора: КРАСНЫЙ, ЖЁЛТЫЙ, ЗЕЛЁНЫЙ или НЕ_ВИДНО
3. Таймер под светофором: цифры показывающие секунды до переключения (если видно)

Ответь СТРОГО в формате:
МОСТ: <статус>
СВЕТОФОР: <цвет>
ТАЙМЕР: <число секунд или НЕ_ВИДНО>

Если что-то не можешь определить — пиши НЕИЗВЕСТНО"""

        response = model.generate_content([prompt, image_part])
        text = response.text.strip().upper()

        # Парсим ответ
        result = {"bridge": "НЕИЗВЕСТНО", "traffic_light": "НЕ_ВИДНО", "timer": None}

        for line in text.split("\n"):
            if "МОСТ:" in line:
                if "СВЕДЁН" in line or "СВЕДЕН" in line:
                    result["bridge"] = "СВЕДЁН"
                elif "РАЗВЕДЁН" in line or "РАЗВЕДЕН" in line:
                    result["bridge"] = "РАЗВЕДЁН"
            elif "СВЕТОФОР:" in line:
                if "КРАСН" in line:
                    result["traffic_light"] = "КРАСНЫЙ"
                elif "ЖЁЛТ" in line or "ЖЕЛТ" in line:
                    result["traffic_light"] = "ЖЁЛТЫЙ"
                elif "ЗЕЛЁН" in line or "ЗЕЛЕН" in line:
                    result["traffic_light"] = "ЗЕЛЁНЫЙ"
            elif "ТАЙМЕР:" in line:
                # Извлекаем число из строки
                numbers = re.findall(r'\d+', line)
                if numbers:
                    result["timer"] = int(numbers[0])

        return result

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        if "quota" in str(e).lower() or "rate limit" in str(e).lower() or "429" in str(e).lower():
            return {"error": "Превышена квота API"}
        return {"error": str(e)[:50]}


def format_status(result: dict, now: str) -> str:
    """Форматирует статус для отправки пользователю."""
    if "error" in result:
        return f"⚠️ Ошибка: {result['error']}\n🕐 {now}"

    bridge = result.get("bridge", "НЕИЗВЕСТНО")
    light = result.get("traffic_light", "НЕ_ВИДНО")
    timer = result.get("timer")

    # Эмодзи для моста
    if bridge == "СВЕДЁН":
        bridge_line = "🟢 Мост СВЕДЁН — проезд открыт"
    elif bridge == "РАЗВЕДЁН":
        bridge_line = "🔴 Мост РАЗВЕДЁН — проезд закрыт"
    else:
        bridge_line = "⚪ Мост: статус неизвестен"

    # Эмодзи для светофора
    light_emoji = {"КРАСНЫЙ": "🔴", "ЖЁЛТЫЙ": "🟡", "ЗЕЛЁНЫЙ": "🟢"}.get(light, "⚫")
    if light == "НЕ_ВИДНО":
        light_line = "🚦 Светофор: не видно"
    elif timer:
        light_line = f"🚦 Светофор: {light_emoji} {light} ({timer} сек)"
    else:
        light_line = f"🚦 Светофор: {light_emoji} {light}"

    return f"{bridge_line}\n{light_line}\n🕐 {now}"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start."""
    # Инлайн-кнопки (работают и в группах)
    inline_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🌉 Проверить мост", callback_data="check_status")],
        [
            InlineKeyboardButton("🔔 Подписаться", callback_data="subscribe"),
            InlineKeyboardButton("🔕 Отписаться", callback_data="unsubscribe"),
        ],
    ])

    await update.message.reply_text(
        "🌉 Бот мониторинга Митяевского моста\n\n"
        "Нажмите кнопку или используйте команды:\n"
        "/status — текущий статус моста\n"
        "/subscribe — подписаться на уведомления\n"
        "/unsubscribe — отписаться",
        reply_markup=inline_keyboard
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /status — показывает текущий статус моста."""
    msg = await update.message.reply_text("📷 Получаю кадр с камеры...")

    # Захватываем кадр
    image_bytes = capture_frame()
    if not image_bytes:
        await msg.edit_text("❌ Не удалось получить кадр с камеры")
        return

    await msg.edit_text("🤖 Анализирую изображение...")

    # Анализируем через Gemini
    result = analyze_bridge(image_bytes)
    now = datetime.now().strftime("%H:%M:%S")
    text = format_status(result, now)

    # Удаляем сообщение о статусе
    await msg.delete()

    # Inline-кнопка для обновления
    inline_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить", callback_data="refresh_status")]
    ])

    # Отправляем фото с результатом
    await update.message.reply_photo(
        photo=io.BytesIO(image_bytes),
        caption=text,
        parse_mode=None,  # Отключаем Markdown, чтобы избежать проблем с форматированием
        reply_markup=inline_keyboard
    )


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Подписаться на уведомления об изменении статуса."""
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)
    await update.message.reply_text(
        f"✅ Вы подписаны на уведомления\n"
        f"Проверка каждые {CHECK_INTERVAL} минут"
    )


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отписаться от уведомлений."""
    chat_id = update.effective_chat.id
    subscribers.discard(chat_id)
    await update.message.reply_text("❌ Вы отписаны от уведомлений")


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на inline-кнопки."""
    query = update.callback_query
    await query.answer()  # Убираем "часики" на кнопке

    if query.data in ("check_status", "refresh_status"):
        # Отправляем сообщение о загрузке
        msg = await query.message.reply_text("📷 Получаю кадр с камеры...")

        # Захватываем кадр
        image_bytes = capture_frame()
        if not image_bytes:
            await msg.edit_text("❌ Не удалось получить кадр с камеры")
            return

        await msg.edit_text("🤖 Анализирую изображение...")

        # Анализируем через Gemini
        result = analyze_bridge(image_bytes)
        now = datetime.now().strftime("%H:%M:%S")
        text = format_status(result, now)

        await msg.delete()

        # Inline-кнопка для обновления
        inline_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data="refresh_status")]
        ])

        await query.message.reply_photo(
            photo=io.BytesIO(image_bytes),
            caption=text,
            parse_mode=None,
            reply_markup=inline_keyboard
        )

    elif query.data == "subscribe":
        chat_id = query.message.chat_id
        if chat_id not in subscribers:
            subscribers.add(chat_id)
            await query.message.reply_text(
                f"✅ Подписка оформлена!\n"
                f"Проверка каждые {CHECK_INTERVAL} минут"
            )
        else:
            await query.message.reply_text("Вы уже подписаны")

    elif query.data == "unsubscribe":
        chat_id = query.message.chat_id
        if chat_id in subscribers:
            subscribers.discard(chat_id)
            await query.message.reply_text("❌ Вы отписаны от уведомлений")
        else:
            await query.message.reply_text("Вы не были подписаны")


async def check_bridge_status(context: ContextTypes.DEFAULT_TYPE):
    """Фоновая задача проверки статуса моста."""
    global last_status

    logger.info("Checking bridge status...")

    image_bytes = capture_frame()
    if not image_bytes:
        logger.warning("Failed to capture frame for monitoring")
        return

    result = analyze_bridge(image_bytes)

    # Пропускаем если ошибка
    if "error" in result:
        logger.warning(f"Analysis error: {result['error']}")
        return

    bridge = result.get("bridge", "НЕИЗВЕСТНО")
    light = result.get("traffic_light", "НЕ_ВИДНО")

    # Проверяем изменения
    if last_status is not None:
        old_bridge = last_status.get("bridge")
        old_light = last_status.get("traffic_light")

        # Уведомляем если изменился мост или светофор
        bridge_changed = old_bridge != bridge and bridge != "НЕИЗВЕСТНО"
        light_changed = old_light != light and light != "НЕ_ВИДНО" and old_light != "НЕ_ВИДНО"

        if bridge_changed or light_changed:
            now = datetime.now().strftime("%H:%M:%S")
            text = "🚨 Изменение!\n" + format_status(result, now)

            for chat_id in subscribers:
                try:
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=io.BytesIO(image_bytes),
                        caption=text,
                        parse_mode=None
                    )
                except Exception as e:
                    logger.error(f"Failed to notify {chat_id}: {e}")

    last_status = result
    logger.info(f"Bridge: {bridge}, Light: {light}")


def main():
    """Запуск бота."""
    # Проверяем наличие всех ключей
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")

    # Создаём приложение с поддержкой JobQueue
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрируем команды
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))

    # Обработчик inline-кнопок
    app.add_handler(CallbackQueryHandler(handle_callback))
    
    # Устанавливаем команды бота для отображения в меню
    from telegram import BotCommand
    commands = [
        BotCommand("start", "Начать работу с ботом"),
        BotCommand("status", "Проверить статус моста"),
        BotCommand("subscribe", "Подписаться на уведомления"),
        BotCommand("unsubscribe", "Отписаться от уведомлений"),
    ]
    
    # Регистрируем команды после запуска приложения
    async def setup_bot_commands(application):
        await application.bot.set_my_commands(commands)
        logger.info("Команды бота зарегистрированы")
    
    # Добавляем задачу на установку команд
    app.job_queue.run_once(setup_bot_commands, when=1)
    
    logger.info("Задача регистрации команд добавлена")

    # Добавляем фоновую задачу мониторинга
    if app.job_queue:
        app.job_queue.run_repeating(
            check_bridge_status,
            interval=CHECK_INTERVAL * 60,  # минуты -> секунды
            first=10  # первая проверка через 10 секунд
        )
    else:
        logger.warning("JobQueue не доступен. Фоновый мониторинг не будет работать.")

    logger.info(f"Bot started. Monitoring every {CHECK_INTERVAL} minutes.")

    # Запускаем
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
