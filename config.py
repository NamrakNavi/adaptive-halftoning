import os
import sys

class Config:
    # Конфигурация для Windows
    
    def __init__(self):
        # Текущая директория проекта
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Пути для Windows (используем относительные пути)
        self.DATASET_PATH = os.path.join(self.BASE_DIR, "halftoning_dataset")
        self.MODELS_DIR = os.path.join(self.BASE_DIR, "models", "saved")
        self.MODEL_SAVE_PATH = os.path.join(self.MODELS_DIR, "adaptive_halftoning_model.pth")
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "results")
        
        # Параметры модели
        self.PATCH_SIZE = 8
        self.BATCH_SIZE = 32
        self.MIN_BATCH_SIZE = 4
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.001
        
        # Параметры изображений
        self.IMG_SIZE = (256, 256)
        
        # Поддерживаемые форматы изображений
        self.IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        # Создаем необходимые директории
        self._create_directories()
    
    def _create_directories(self):
        """Создание всех необходимых директорий"""
        directories = [
            self.MODELS_DIR,
            self.OUTPUT_DIR,
            os.path.join(self.DATASET_PATH, 'train', 'colored', 'nature'),
            os.path.join(self.DATASET_PATH, 'train', 'colored', 'transport'),
            os.path.join(self.DATASET_PATH, 'train', 'black and white', 'nature'),
            os.path.join(self.DATASET_PATH, 'train', 'black and white', 'transport'),
            os.path.join(self.DATASET_PATH, 'test', 'colored', 'nature'),
            os.path.join(self.DATASET_PATH, 'test', 'colored', 'transport'),
            os.path.join(self.DATASET_PATH, 'test', 'black and white', 'nature'),
            os.path.join(self.DATASET_PATH, 'test', 'black and white', 'transport'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def update_dataset_path(self, new_path):
        """Обновление пути к датасету"""
        self.DATASET_PATH = new_path
        print(f"Путь к датасету обновлен: {new_path}")
    
    def update_output_dir(self, new_dir):
        """Обновление директории для результатов"""
        self.OUTPUT_DIR = new_dir
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        print(f"Директория для результатов обновлена: {new_dir}")