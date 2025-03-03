import os
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger
import requests
from bs4 import BeautifulSoup
import time
from config import *

class BridgeDetector:
    def __init__(self):
        """Инициализация детектора мостов"""
        logger.info("Инициализация BridgeDetector")
        self._ensure_directories()
        self.model = self._load_model()
        self._log_model_info()

    def _ensure_directories(self):
        """Создание необходимых директорий"""
        for directory in [SCREENSHOTS_DIR, OUTPUT_DIR, MODELS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Создана директория: {directory}")

    def _load_model(self):
        """Загрузка модели YOLO"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")
        
        logger.info(f"Загрузка модели из: {MODEL_PATH}")
        return YOLO(MODEL_PATH)

    def _log_model_info(self):
        """Логирование информации о модели"""
        try:
            model_info = self.model.info()
            logger.info(f"Информация о модели:")
            logger.info(f"- Имя модели: {model_info.name if hasattr(model_info, 'name') else 'Неизвестно'}")
            logger.info(f"- Версия: {model_info.version if hasattr(model_info, 'version') else 'Неизвестно'}")
            logger.info(f"- Количество классов: {model_info.nc if hasattr(model_info, 'nc') else 'Неизвестно'}")
            logger.info(f"- Классы: {model_info.names if hasattr(model_info, 'names') else 'Неизвестно'}")
        except Exception as e:
            logger.warning(f"Не удалось получить информацию о модели: {str(e)}")

    def _get_class_name(self, class_id):
        """Получение имени класса по его ID"""
        try:
            return CLASSES[class_id]
        except IndexError:
            logger.warning(f"Обнаружен неизвестный класс с ID: {class_id}")
            return f"unknown_class_{class_id}"

    def _get_class_color(self, class_name):
        """Получение цвета для класса"""
        return CLASS_COLORS.get(class_name, (0, 0, 255))  # Красный цвет по умолчанию

    def _determine_bridge_status(self, detections, image_shape):
        """
        Определение статуса моста на основе детекций
        
        Args:
            detections (list): Список детекций
            image_shape (tuple): Размеры изображения (height, width)
            
        Returns:
            str: Статус моста ('open', 'closed', 'unknown')
        """
        if not detections:
            return 'unknown'
            
        # Получаем все детекции моста
        bridge_detections = [d for d in detections if d['class'] == 'bridge']
        if not bridge_detections:
            return 'unknown'
            
        # Анализируем размеры и положение моста
        for det in bridge_detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            image_area = image_shape[0] * image_shape[1]
            area_ratio = area / image_area
            
            # Если мост занимает большую часть изображения, вероятно он закрыт
            if area_ratio > 0.5:  # Уменьшаем порог до 50% площади изображения
                return 'closed'
            # Если мост занимает меньшую часть и находится в верхней части изображения
            elif area_ratio < 0.4 and y1 < image_shape[0] * 0.4:  # Увеличиваем порог до 40% площади и верхней части
                return 'open'
            
            # Добавляем дополнительную проверку на положение моста
            center_y = (y1 + y2) / 2
            if center_y < image_shape[0] * 0.3:  # Если центр моста в верхней трети
                return 'open'
            elif center_y > image_shape[0] * 0.7:  # Если центр моста в нижней трети
                return 'closed'
                
        return 'unknown'

    def detect_image(self, image_path, save_result=True):
        """
        Детекция мостов на изображении
        
        Args:
            image_path (str): Путь к изображению
            save_result (bool): Сохранять ли результат
            
        Returns:
            tuple: (list: Результаты детекции, str: Статус моста)
        """
        logger.info(f"Обработка изображения: {image_path}")
        try:
            # Загружаем изображение для получения размеров
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
            # Получаем предсказания
            results = self.model.predict(
                source=image_path,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )[0]
            
            # Преобразуем результаты в удобный формат
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self._get_class_name(class_id)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
                
                logger.debug(f"Обнаружен объект: {class_name} (уверенность: {confidence:.2f})")
            
            # Определяем статус моста
            bridge_status = self._determine_bridge_status(detections, img.shape[:2])
            logger.info(f"Статус моста: {bridge_status}")
            
            if save_result:
                # Создаем копию изображения для визуализации
                self._draw_detections(img, detections)
                
                # Добавляем текст со статусом
                status_text = f"Bridge Status: {bridge_status.upper()}"
                cv2.putText(img, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Сохраняем результат
                output_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(image_path)}")
                cv2.imwrite(output_path, img)
                logger.info(f"Результат сохранен в: {output_path}")
            
            return detections, bridge_status
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {str(e)}")
            raise

    def _draw_detections(self, image, detections):
        """Отрисовка детекций на изображении"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_name = det['class']
            color = self._get_class_color(class_name)
            
            # Рисуем прямоугольник
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Добавляем текст с классом и уверенностью
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_video(self, input_path, output_path):
        """
        Обработка видео для детекции мостов
        
        Args:
            input_path (str): Путь к входному видео
            output_path (str): Путь для сохранения результата
        """
        logger.info(f"Начало обработки видео: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видео файл")

        # Получаем параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
        out = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (width, height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Получаем предсказания для текущего кадра
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
                    class_name = self._get_class_name(class_id)
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
                
                # Определяем статус моста
                bridge_status = self._determine_bridge_status(detections, frame.shape[:2])
                
                # Отрисовываем детекции на кадре
                self._draw_detections(frame, detections)
                
                # Добавляем текст со статусом
                status_text = f"Bridge Status: {bridge_status.upper()}"
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Записываем обработанный кадр
                out.write(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Обработано кадров: {frame_count}, Статус моста: {bridge_status}")

        finally:
            cap.release()
            out.release()
            logger.info(f"Обработка видео завершена. Всего кадров: {frame_count}")

    def process_webcam(self, url, output_path=None, duration=None):
        """
        Обработка потока с веб-камеры
        
        Args:
            url (str): URL потока веб-камеры
            output_path (str, optional): Путь для сохранения видео
            duration (int, optional): Длительность записи в секундах
        """
        logger.info(f"Подключение к веб-камере: {url}")
        
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise ValueError("Не удалось подключиться к веб-камере")

        # Получаем параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Создаем VideoWriter если указан путь для сохранения
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Запись видео в: {output_path}")

        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Ошибка получения кадра")
                    break

                # Проверяем длительность записи
                if duration and (time.time() - start_time) > duration:
                    logger.info(f"Достигнута максимальная длительность записи: {duration} секунд")
                    break

                # Получаем предсказания для текущего кадра
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
                    class_name = self._get_class_name(class_id)
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
                
                # Определяем статус моста
                bridge_status = self._determine_bridge_status(detections, frame.shape[:2])
                
                # Отрисовываем детекции на кадре
                self._draw_detections(frame, detections)
                
                # Добавляем текст со статусом
                status_text = f"Bridge Status: {bridge_status.upper()}"
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Показываем кадр
                cv2.imshow('Bridge Detection', frame)
                
                # Записываем кадр если указан путь для сохранения
                if out:
                    out.write(frame)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Обработано кадров: {frame_count}, Статус моста: {bridge_status}")
                
                # Выход по клавише 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            logger.info(f"Обработка потока завершена. Всего кадров: {frame_count}")

def get_camera_urls():
    """
    Получение URL камер с сайта
    """
    url = "https://inkotel.ru/veb-kamery/"
    try:
        # Отключаем проверку SSL-сертификата
        response = requests.get(url, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Ищем все iframe с трансляциями
        iframes = soup.find_all('iframe')
        camera_urls = []
        
        for iframe in iframes:
            src = iframe.get('src', '')
            if src:
                camera_urls.append(src)
                logger.info(f"Найдена камера: {src}")
        
        return camera_urls
    except Exception as e:
        logger.error(f"Ошибка при получении URL камер: {str(e)}")
        return []

def main():
    """Основная функция для демонстрации работы детектора"""
    detector = BridgeDetector()
    
    # Используем прямой RTSP URL
  
    camera_url = "https://node007.youlook.ru/cam001544/index.m3u8?token=f2d6d079b8fc54a745e085b0c34e2e08"

    logger.info(f"Начинаем обработку камеры: {camera_url}")
    
    try:
        # Запускаем обработку потока
        detector.process_webcam(
            camera_url,
            output_path="output/webcam_output.mp4",
            duration=3600  # 1 час
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке потока: {str(e)}")

if __name__ == "__main__":
    # Отключаем предупреждения о небезопасном SSL
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    main() 