# Инструкция по деплою 🚀

## Подготовка сервера

1. **Создание пользователя:**
```bash
# Создаем пользователя bridge_bot
sudo adduser bridge_bot
sudo usermod -aG sudo bridge_bot
```

2. **Настройка SSH:**
```bash
# Переключаемся на нового пользователя
su - bridge_bot

# Создаем SSH ключи
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "bridge_bot@server"

# Показываем публичный ключ для добавления в GitHub
cat ~/.ssh/id_ed25519.pub
```

3. **Установка зависимостей:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip git
```

4. **Создание директории проекта:**
```bash
sudo mkdir -p /opt/bridge-status-bot
sudo chown bridge_bot:bridge_bot /opt/bridge-status-bot
```

## Настройка GitHub

1. **Добавить секреты в репозиторий:**
   - `GH_TOKEN`: Personal Access Token
   - `SERVER_HOST`: IP адрес сервера
   - `SERVER_USER`: bridge_bot
   - `SSH_PRIVATE_KEY`: содержимое ~/.ssh/id_ed25519
   - `TELEGRAM_TOKEN`: токен Telegram бота
   - `CAMERA_URL`: URL камеры

2. **Настроить окружение production:**
   - Settings -> Environments -> New environment
   - Название: production
   - Добавить те же секреты

## Настройка systemd

1. **Создать файл сервиса:**
```bash
sudo nano /etc/systemd/system/bridge_bot.service
```

2. **Содержимое файла:**
```ini
[Unit]
Description=Bridge Status Detection Bot
After=network.target

[Service]
Type=simple
User=bridge_bot
WorkingDirectory=/opt/bridge-status-bot
Environment=PYTHONPATH=/opt/bridge-status-bot
ExecStart=/opt/bridge-status-bot/.venv/bin/python telegram_bot_v2.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. **Активация сервиса:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable bridge_bot
sudo systemctl start bridge_bot
```

## Первичный деплой

1. **Клонирование репозитория:**
```bash
cd /opt/bridge-status-bot
git clone https://github.com/dvorobiev/bridge-status-bot.git .
```

2. **Настройка окружения:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Настройка переменных окружения:**
```bash
cp .env.example .env
nano .env  # Добавить TELEGRAM_TOKEN и CAMERA_URL
```

## Проверка работоспособности

1. **Проверка статуса:**
```bash
sudo systemctl status bridge_bot
```

2. **Просмотр логов:**
```bash
sudo journalctl -u bridge_bot -f
```

## Обновление

Обновление происходит автоматически при пуше в ветку main через GitHub Actions.

## Откат изменений

В случае проблем:
```bash
# Остановить сервис
sudo systemctl stop bridge_bot

# Откатить к последней рабочей версии
git reset --hard HEAD^
git pull origin main

# Перезапустить сервис
sudo systemctl start bridge_bot
``` 