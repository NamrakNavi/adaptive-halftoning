import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.file_utils import find_images_in_folder

class HalftoningDataset(Dataset):
    # Датасет для адаптивного растрирования (Windows)
    
    # Поддерживаемые форматы изображений
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    def __init__(self, root_dir, mode='train', color_type='all', category='all',
                 transform=None, target_size=(256, 256)):
        """
        Args:
            root_dir: путь к корневой папке датасета
            mode: 'train' или 'test'
            color_type: 'all', 'colored', 'black and white'
            category: 'all', 'nature', 'transport'
            transform: трансформации для изображений
            target_size: целевой размер изображений
        """
        self.root_dir = root_dir
        self.mode = mode
        self.color_type = color_type
        self.category = category
        self.transform = transform
        self.target_size = target_size
        
        self.image_paths = []
        self._load_image_paths()
        
        print(f"\nДатасет: {mode.upper()}")
        print(f"Цвет: {color_type}, Категория: {category}")
        print(f"Загружено изображений: {len(self.image_paths)}")
        
        if len(self.image_paths) == 0:
            print("⚠ ВНИМАНИЕ: Не найдено изображений!")
            print(f"  Путь: {os.path.join(root_dir, mode)}")

    def _load_image_paths(self):
        # Загрузка путей к изображениям
        base_path = os.path.join(self.root_dir, self.mode)
        
        if not os.path.exists(base_path):
            print(f"Предупреждение: Директория {base_path} не существует")
            return

        # Определяем какие директории обходить
        if self.color_type == 'all':
            color_types = ['colored', 'black and white']
        else:
            color_types = [self.color_type]

        if self.category == 'all':
            categories = ['nature', 'transport']
        else:
            categories = [self.category]

        # Поиск изображений
        for color_type in color_types:
            color_path = os.path.join(base_path, color_type)
            
            if not os.path.exists(color_path):
                continue

            for category in categories:
                category_path = os.path.join(color_path, category)
                
                if os.path.exists(category_path):
                    # Ищем изображения в папке категории
                    images = find_images_in_folder(category_path, self.IMAGE_EXTENSIONS)
                    self.image_paths.extend(images)
                else:
                    # Если нет структуры с категориями, ищем прямо в color_path
                    images = find_images_in_folder(color_path, self.IMAGE_EXTENSIONS)
                    self.image_paths.extend(images)

        # Если всё ещё нет изображений, ищем во всей папке mode
        if len(self.image_paths) == 0:
            print("Поиск изображений без фильтрации по цвету/категории...")
            self.image_paths = find_images_in_folder(base_path, self.IMAGE_EXTENSIONS)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Загрузка изображения
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

            # Преобразование в тензор
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Стандартная нормализация
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                image_tensor = transform(image)

            if self.mode == 'train':
                # Создаем целевое изображение
                target = self._create_target_image(image)
                return image_tensor, target
            else:
                return image_tensor

        except Exception as e:
            print(f"Ошибка при загрузке {img_path}: {e}")
            # Возвращаем пустое изображение в случае ошибки
            dummy_image = torch.zeros((3, *self.target_size))
            if self.mode == 'train':
                target = torch.zeros((1, *self.target_size))
                return dummy_image, target
            return dummy_image

    def _create_target_image(self, image):
        # Создание целевого изображения с помощью error diffusion
        # Конвертируем в numpy
        image_np = np.array(image) / 255.0

        # Преобразуем в оттенки серого
        if len(image_np.shape) == 3:
            gray_image = 0.299 * image_np[:,:,0] + 0.587 * image_np[:,:,1] + 0.114 * image_np[:,:,2]
        else:
            gray_image = image_np

        h, w = gray_image.shape
        img_float = gray_image.copy()
        target_np = np.zeros((h, w))

        # Floyd-Steinberg error diffusion
        for i in range(h):
            for j in range(w):
                old_pixel = img_float[i, j]
                new_pixel = 1.0 if old_pixel > 0.5 else 0.0
                target_np[i, j] = new_pixel
                quant_error = old_pixel - new_pixel

                # Распространение ошибки
                if j + 1 < w:
                    img_float[i, j+1] += quant_error * 7/16
                if i + 1 < h and j - 1 >= 0:
                    img_float[i+1, j-1] += quant_error * 3/16
                if i + 1 < h:
                    img_float[i+1, j] += quant_error * 5/16
                if i + 1 < h and j + 1 < w:
                    img_float[i+1, j+1] += quant_error * 1/16

        return torch.FloatTensor(target_np).unsqueeze(0)