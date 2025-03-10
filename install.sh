#!/bin/bash

# Проверяем, запущен ли скрипт с правами root
if [ "$EUID" -ne 0 ]; then 
    echo "Пожалуйста, запустите скрипт с правами root"
    exit 1
fi

# Установка необходимых пакетов
apt-get update
apt-get install -y python3-venv python3-pip git jq

# Создание директории для бота
mkdir -p /opt/bridge-status-bot
cd /opt/bridge-status-bot

# Клонирование репозитория
git clone https://github.com/dvorobiev/bridge-status-bot.git .

# Создание виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Создание .env файла из GitHub Secrets
if [ -n "$GITHUB_TOKEN" ]; then
    echo "Получение секретов из GitHub..."
    
    # Получаем секреты через GitHub API
    TELEGRAM_TOKEN=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/dvorobiev/bridge-status-bot/actions/secrets/public-key" | jq -r '.key')
    
    CAMERA_URL=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/dvorobiev/bridge-status-bot/actions/secrets/public-key" | jq -r '.key')
    
    # Создаем .env файл
    cat > .env << EOL
# Конфигурация бота
TELEGRAM_TOKEN=$TELEGRAM_TOKEN
CAMERA_URL=$CAMERA_URL

# Параметры детекции
CONFIDENCE_THRESHOLD=0.4
IOU_THRESHOLD=0.3
EOL
else
    # Если нет доступа к GitHub Secrets, используем пример
    cp .env.example .env
    echo "Создан файл .env из примера. Пожалуйста, отредактируйте его и добавьте ваши настройки."
    exit 1
fi

# Копирование systemd сервиса
cp bridge_bot.service /etc/systemd/system/

# Перезагрузка systemd и запуск сервиса
systemctl daemon-reload
systemctl enable bridge_bot
systemctl start bridge_bot

echo "Установка завершена! Проверьте статус бота командой: systemctl status bridge_bot" 