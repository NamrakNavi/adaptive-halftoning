import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from models.unet import UNet
from algorithms.halftoning_algorithms import HalftoningAlgorithms
from data.dataset import HalftoningDataset
from utils.visualization import plot_results, plot_training_loss
from utils.file_utils import ensure_dir


class AdaptiveHalftoningSystem:
    def __init__(self, dataset_path, config):
        self.config = config
        self.config.DATASET_PATH = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.BCEWithLogitsLoss()
        self.halftoner = HalftoningAlgorithms(config)
        ensure_dir(config.OUTPUT_DIR)
        print(f"Используется устройство: {self.device}")

    def create_dataloaders(self, train_color='all', train_category='all', test_color='all', test_category='all', transform=None):
        from torchvision import transforms
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(self.config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        train_dataset = HalftoningDataset(
            self.config.DATASET_PATH,
            mode='train', color_type=train_color, category=train_category,
            transform=transform, target_size=self.config.IMG_SIZE
        )
        test_dataset = HalftoningDataset(
            self.config.DATASET_PATH,
            mode='test', color_type=test_color, category=test_category,
            transform=transform, target_size=self.config.IMG_SIZE
        )

        if len(train_dataset) == 0:
            print('ОШИБКА: Тренировочный датасет пуст!')
            return None, None

        batch_size = self._get_optimal_batch_size(len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = None
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False, num_workers=2)
        return train_loader, test_loader

    def _get_optimal_batch_size(self, dataset_size):
        if dataset_size < self.config.MIN_BATCH_SIZE:
            return max(1, dataset_size)
        if dataset_size < self.config.BATCH_SIZE:
            return max(self.config.MIN_BATCH_SIZE, min(dataset_size, self.config.BATCH_SIZE))
        return self.config.BATCH_SIZE

    def train(self, train_loader, epochs=None):
        if train_loader is None or len(train_loader) == 0:
            print('Невозможно обучить модель: нет данных')
            return
        if epochs is None:
            epochs = self.config.EPOCHS

        self.model.train()
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                if batch_idx % max(1, len(train_loader) // 5) == 0:
                    print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                losses.append(avg_loss)
                print(f'Epoch {epoch + 1} завершен. Средняя loss: {avg_loss:.6f}')
        self.save_model()
        if losses:
            plot_training_loss(losses, self.config.OUTPUT_DIR)

    def save_model(self, path=None):
        if path is None:
            path = self.config.MODEL_SAVE_PATH
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, path)
        print(f'Модель сохранена: {path}')
        return path

    def load_model(self, path=None):
        if path is None:
            path = self.config.MODEL_SAVE_PATH
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Модель загружена: {path}')
            return True
        print(f'Модель не найдена: {path}')
        return False

    def process_image(self, image_path, save_results=True):
        try:
            from skimage.metrics import structural_similarity as ssim
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image.resize(self.config.IMG_SIZE), dtype=np.float32) / 255.0

            ordered = self.halftoner.ordered_dithering(image_np)
            error_diff = self.halftoner.error_diffusion(image_np)
            adaptive_result = self.halftoner.adaptive_halftoning(image_np, self.model, self.device, return_details=True)
            adaptive = adaptive_result['halftone']
            adaptive_preview = adaptive_result['preview']

            gray_image = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
            ssim_ordered = ssim(gray_image, ordered, data_range=1.0)
            ssim_error = ssim(gray_image, error_diff, data_range=1.0)
            ssim_adaptive = ssim(gray_image, adaptive, data_range=1.0)
            ssim_preview = ssim(gray_image, adaptive_preview, data_range=1.0)

            if save_results:
                plot_results(
                    image_np, ordered, error_diff, adaptive, adaptive_preview,
                    ssim_ordered, ssim_error, ssim_adaptive, ssim_preview,
                    os.path.basename(image_path), self.config.OUTPUT_DIR
                )
                self._save_single_outputs(image_path, ordered, error_diff, adaptive, adaptive_preview)

            return {
                'ssim_ordered': ssim_ordered,
                'ssim_error': ssim_error,
                'ssim_adaptive': ssim_adaptive,
                'ssim_adaptive_preview': ssim_preview,
                'image_name': os.path.basename(image_path)
            }
        except Exception as e:
            print(f'Ошибка при обработке {image_path}: {e}')
            return None

    def _save_single_outputs(self, image_path, ordered, error_diff, adaptive, adaptive_preview):
        from PIL import Image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        Image.fromarray((ordered * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_ordered.png'))
        Image.fromarray((error_diff * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_error_diffusion.png'))
        Image.fromarray((adaptive * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_adaptive.png'))
        Image.fromarray((adaptive_preview * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_stable_preview.png'))
