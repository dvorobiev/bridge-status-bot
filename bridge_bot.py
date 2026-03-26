#!/usr/bin/env python3
"""
Telegram-бот для мониторинга понтонного моста.
Использует Gemini Vision для определения статуса: СВЕДЁН или РАЗВЕДЁН.
"""

import os
import io
import logging
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import httpx
import re

import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Загружаем переменные окружения
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Настройка Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Модели от продвинутой к простой (fallback при исчерпании квоты)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]

# Настройки камеры rtsp.me (Митяевский мост)
RTSP_ME_EMBED_URL = "https://rtsp.me/embed/yEDF9iDT/"


def get_rtspme_stream_url() -> tuple[str | None, str | None]:
    """Получает актуальный m3u8 URL из rtsp.me embed страницы.
    Возвращает (url, error_message)."""
    try:
        with httpx.Client(timeout=15, verify=False) as client:
            response = client.get(RTSP_ME_EMBED_URL, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code != 200:
                return None, f"HTTP {response.status_code} при обращении к rtsp.me"
            match = re.search(r'https://msk\.rtsp\.me/[^"\']+\.m3u8[^"\']*', response.text)
            if match:
                return match.group(0), None
            return None, "Ссылка на поток не найдена на странице rtsp.me"
    except httpx.TimeoutException:
        return None, "Таймаут подключения к rtsp.me (15с)"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def capture_frame() -> tuple[bytes | None, str | None]:
    """Захватывает кадр с камеры rtsp.me через ffmpeg.
    Возвращает (image_bytes, error_message)."""
    stream_url, err = get_rtspme_stream_url()
    if not stream_url:
        logger.error(f"Failed to get stream URL: {err}")
        return None, err or "Не удалось получить URL потока"

    logger.info(f"Capturing frame from: {stream_url[:60]}...")

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
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode == 0 and result.stdout:
            logger.info(f"Frame captured: {len(result.stdout)} bytes")
            return result.stdout, None

        stderr = result.stderr.decode(errors="replace")[:300]
        logger.error(f"ffmpeg error: {stderr}")
        return None, f"ffmpeg код {result.returncode}: {stderr}"

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout")
        return None, "Таймаут захвата кадра ffmpeg (30с)"
    except Exception as e:
        logger.error(f"Capture error: {e}")
        return None, f"{type(e).__name__}: {e}"


def analyze_bridge(image_bytes: bytes) -> dict:
    """Отправляет изображение в Gemini и получает статус моста и светофора."""
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }

    prompt = """Изображение с камеры понтонного моста. Справа виден светофор с цифровым табло.

Определи:
1. Статус моста: СВЕДЁН (цельный, можно ехать) или РАЗВЕДЁН (есть разрыв)
2. Цвет светофора: КРАСНЫЙ, ЖЁЛТЫЙ или ЗЕЛЁНЫЙ
3. Цифры на табло светофора (секунды обратного отсчёта) — внимательно прочитай число

Формат ответа:
МОСТ: СВЕДЁН или РАЗВЕДЁН
СВЕТОФОР: цвет
ТАЙМЕР: число (только цифры, например 45 или 120)"""

    # Пробуем модели от продвинутой к простой
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image_part])
            text = response.text.strip().upper()

            logger.info(f"Used model: {model_name}")

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
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        result["timer"] = int(numbers[0])

            return result

        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "rate limit" in err_str or "429" in err_str:
                logger.warning(f"{model_name}: quota exceeded, trying next...")
                continue
            else:
                logger.error(f"Gemini error ({model_name}): {e}")
                return {"error": str(e)[:50]}

    return {"error": "Все модели исчерпали квоту"}


def format_status(result: dict, now: str) -> str:
    """Форматирует статус для отправки пользователю."""
    if "error" in result:
        return f"⚠️ Ошибка: {result['error']}\n🕐 {now}"

    bridge = result.get("bridge", "НЕИЗВЕСТНО")
    light = result.get("traffic_light", "НЕ_ВИДНО")
    timer = result.get("timer")

    if bridge == "СВЕДЁН":
        bridge_line = "🟢 Мост СВЕДЁН — проезд открыт"
    elif bridge == "РАЗВЕДЁН":
        bridge_line = "🔴 Мост РАЗВЕДЁН — проезд закрыт"
    else:
        bridge_line = "⚪ Мост: статус неизвестен"

    light_emoji = {"КРАСНЫЙ": "🔴", "ЖЁЛТЫЙ": "🟡", "ЗЕЛЁНЫЙ": "🟢"}.get(light, "⚫")
    if light == "НЕ_ВИДНО":
        light_line = "🚦 Светофор: ?"
    elif timer:
        light_line = f"🚦 Светофор: {light_emoji} {timer} сек"
    else:
        light_line = f"🚦 Светофор: {light_emoji}"

    return f"{bridge_line}\n{light_line}\n🕐 {now}"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start."""
    inline_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🌉 Проверить мост", callback_data="check_status")],
    ])

    await update.message.reply_text(
        "🌉 Бот мониторинга Митяевского моста\n\n"
        "Нажмите кнопку или команду /status",
        reply_markup=inline_keyboard
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /status — показывает текущий статус моста."""
    msg = await update.message.reply_text("📷 Получаю кадр с камеры...")

    image_bytes, capture_error = capture_frame()
    if not image_bytes:
        now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        await msg.edit_text(
            f"❌ Не удалось получить кадр с камеры\n"
            f"📷 Источник: {RTSP_ME_EMBED_URL}\n"
            f"🕐 Время: {now}\n"
            f"⚠️ Причина: {capture_error}"
        )
        return

    await msg.edit_text("🤖 Анализирую изображение...")

    result = analyze_bridge(image_bytes)
    now = datetime.now().strftime("%H:%M:%S")
    text = format_status(result, now)

    await msg.delete()

    inline_keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить", callback_data="refresh_status")]
    ])

    await update.message.reply_photo(
        photo=io.BytesIO(image_bytes),
        caption=text,
        parse_mode=None,
        reply_markup=inline_keyboard
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий на inline-кнопки."""
    query = update.callback_query
    await query.answer()

    if query.data in ("check_status", "refresh_status"):
        msg = await query.message.reply_text("📷 Получаю кадр с камеры...")

        image_bytes, capture_error = capture_frame()
        if not image_bytes:
            now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            await msg.edit_text(
                f"❌ Не удалось получить кадр с камеры\n"
                f"📷 Источник: {RTSP_ME_EMBED_URL}\n"
                f"🕐 Время: {now}\n"
                f"⚠️ Причина: {capture_error}"
            )
            return

        await msg.edit_text("🤖 Анализирую изображение...")

        result = analyze_bridge(image_bytes)
        now = datetime.now().strftime("%H:%M:%S")
        text = format_status(result, now)

        await msg.delete()

        inline_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data="refresh_status")]
        ])

        await query.message.reply_photo(
            photo=io.BytesIO(image_bytes),
            caption=text,
            parse_mode=None,
            reply_markup=inline_keyboard
        )


def main():
    """Запуск бота."""
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CallbackQueryHandler(handle_callback))

    commands = [
        BotCommand("start", "Начать работу с ботом"),
        BotCommand("status", "Проверить статус моста"),
    ]

    async def setup_bot_commands(application):
        await application.bot.set_my_commands(commands)
        logger.info("Команды бота зарегистрированы")

    app.job_queue.run_once(setup_bot_commands, when=1)

    logger.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
