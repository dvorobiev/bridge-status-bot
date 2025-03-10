#!/bin/bash

# Проверяем, запущен ли скрипт с правами root
if [ "$EUID" -ne 0 ]; then 
    echo "Пожалуйста, запустите скрипт с правами root"
    exit 1
fi

# Установка необходимых пакетов
apt-get update
apt-get install -y python3-venv python3-pip git

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

# Создание .env файла из примера, если он не существует
if [ ! -f .env ]; then
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