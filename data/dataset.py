import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.file_utils import find_images_in_folder


class HalftoningDataset(Dataset):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

    def __init__(self, root_dir, mode='train', color_type='all', category='all', transform=None, target_size=(256, 256)):
        self.root_dir = root_dir
        self.mode = mode
        self.color_type = color_type
        self.category = category
        self.transform = transform
        self.target_size = target_size
        self.image_paths = []
        self._load_image_paths()

        print(f'Датасет: {mode.upper()}')
        print(f'Цвет: {color_type}, Категория: {category}')
        print(f'Загружено изображений: {len(self.image_paths)}')

    def _load_image_paths(self):
        base_path = os.path.join(self.root_dir, self.mode)
        if not os.path.exists(base_path):
            print(f'Предупреждение: Директория {base_path} не существует')
            return

        color_types = ['colored', 'black and white'] if self.color_type == 'all' else [self.color_type]
        categories = ['nature', 'transport'] if self.category == 'all' else [self.category]

        for color_type in color_types:
            color_path = os.path.join(base_path, color_type)
            if not os.path.exists(color_path):
                continue
            for category in categories:
                category_path = os.path.join(color_path, category)
                if os.path.exists(category_path):
                    images = find_images_in_folder(category_path, self.IMAGE_EXTENSIONS)
                    self.image_paths.extend(images)
                else:
                    images = find_images_in_folder(color_path, self.IMAGE_EXTENSIONS)
                    self.image_paths.extend(images)

        if len(self.image_paths) == 0:
            self.image_paths = find_images_in_folder(base_path, self.IMAGE_EXTENSIONS)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

            if self.transform:
                image_tensor = self.transform(image)
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image)

            if self.mode == 'train':
                target = self._create_target_param_map(image)
                return image_tensor, target
            return image_tensor

        except Exception as e:
            print(f'Ошибка при загрузке {img_path}: {e}')
            dummy_image = torch.zeros((3, *self.target_size), dtype=torch.float32)
            if self.mode == 'train':
                target = torch.full((1, *self.target_size), 0.5, dtype=torch.float32)
                return dummy_image, target
            return dummy_image

    def _create_target_param_map(self, image):
        image_np = np.array(image, dtype=np.float32) / 255.0
        gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
        h, w = gray.shape
        patch_size = 8
        target = np.full((h, w), 0.5, dtype=np.float32)

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                i_end = min(i + patch_size, h)
                j_end = min(j + patch_size, w)
                patch = gray[i:i_end, j:j_end]
                if patch.size == 0:
                    continue
                mu = float(np.mean(patch))
                sigma = max(float(np.std(patch)), 1e-3)
                centered = (patch - mu) / (2.0 * sigma)
                p_patch = 0.5 + centered
                p_patch = np.clip(p_patch, 0.0, 1.0)
                target[i:i_end, j:j_end] = p_patch

        return torch.from_numpy(target).unsqueeze(0).float()