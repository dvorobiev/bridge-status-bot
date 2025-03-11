# Bridge Status Detection Bot 🌉

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![Telegram Bot API](https://img.shields.io/badge/Telegram-Bot%20API-blue.svg)](https://core.telegram.org/bots/api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dvorobiev/bridge-status-bot/graphs/commit-activity)

Telegram бот для мониторинга состояния моста с использованием компьютерного зрения и YOLOv8. Автоматически определяет состояние моста (открыт/закрыт) и обучается на основе обратной связи от пользователей.

## ✨ Возможности

- 🔍 Определение текущего состояния моста (открыт/закрыт)
- 🤖 Автоматическое распознавание с помощью YOLOv8
- 📸 Отправка фотографий с результатами распознавания
- 🎯 Режим интерактивного обучения для улучшения точности
- ✅ Валидация предсказаний с сохранением корректных меток
- 🚀 Автоматический деплой через GitHub Actions
- 📊 Сохранение истории обучения и результатов

## 🛠 Технологии

- **YOLOv8**: Современная архитектура для объектной детекции
- **Python 3.8+**: Основной язык разработки
- **OpenCV**: Обработка изображений
- **Telegram Bot API**: Взаимодействие с пользователями
- **GitHub Actions**: CI/CD пайплайн

## 📁 Структура проекта

```
project_root/
├── bridge_detector_v2/           # Основная директория проекта
│   ├── models/                   # Модели YOLOv8
│   │   └── best.pt              # Текущая рабочая модель
│   ├── dataset/                  # Датасет для обучения
│   │   ├── train/               # Тренировочные данные
│   │   ├── val/                 # Валидационные данные
│   │   └── new_data/            # Новые данные для дообучения
│   ├── output/                   # Результаты обучения
│   │   └── train_YYYYMMDD_HHMMSS/  # Директории с результатами
│   └── temp/                    # Временные файлы
├── config.py                     # Конфигурация проекта
└── telegram_bot_v2.py           # Основной код бота
```

## ⚙️ Установка

### 🐳 Локальная установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/dvorobiev/bridge-status-bot.git
cd bridge-status-bot
```

2. Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv .venv
source .venv/bin/activate  # для Linux/Mac
# или
.venv\Scripts\activate  # для Windows
pip install -r requirements.txt
```

3. Настройте переменные окружения:
```bash
cp .env.example .env
# Отредактируйте .env файл
```

### 🖥 Установка на сервер

1. Создайте Personal Access Token (PAT) на GitHub с правами:
   - `repo` (полный доступ к репозиторию)
   - `workflow` (доступ к GitHub Actions)

2. Выполните установку:
```bash
curl -O https://raw.githubusercontent.com/dvorobiev/bridge-status-bot/main/install.sh
chmod +x install.sh
sudo ./install.sh
```

## 🚀 Использование

1. Запустите бота:
```bash
python telegram_bot_v2.py
```

2. В Telegram найдите бота и отправьте `/start`

3. Доступные команды:
   - 🔍 `/status` - проверить текущий статус моста
   - 📚 `/train` - начать режим обучения
   - 🎓 `/stop_train` - завершить обучение
   - ❓ `/help` - показать справку

## 🎯 Обучение модели

1. **Подготовка данных:**
   - Запустите режим обучения через бота
   - Размечайте получаемые кадры (Открыт/Закрыт)
   - Новые данные сохраняются в `dataset/new_data/`

2. **Процесс обучения:**
   - При завершении режима обучения данные автоматически разделяются на train/val
   - Запускается процесс обучения YOLOv8
   - Результаты сохраняются в `output/train_YYYYMMDD_HHMMSS/`
   - Лучшая модель автоматически копируется в `models/best.pt`

3. **Валидация:**
   - После каждого предсказания пользователь может подтвердить или исправить результат
   - Исправленные примеры автоматически добавляются в датасет

## 🔄 CI/CD

При пуше в ветку `main`:
1. Запускается GitHub Actions workflow
2. Код автоматически деплоится на сервер
3. Перезапускается systemd сервис
4. Применяются новые изменения

## 📝 Лицензия

MIT License. См. файл [LICENSE](LICENSE) для деталей.

## 👥 Участие в разработке

Мы приветствуем ваш вклад в проект! Пожалуйста:
1. 🍴 Форкните репозиторий
2. 🔧 Создайте ветку для ваших изменений
3. 📝 Внесите изменения и закоммитьте их
4. 📫 Создайте Pull Request

## 📞 Поддержка

Если у вас возникли вопросы или проблемы:
1. 📝 Создайте Issue в репозитории
2. 📧 Свяжитесь с разработчиками
3. 🌟 Поставьте звезду проекту, если он вам помог 