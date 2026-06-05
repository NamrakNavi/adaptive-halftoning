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
        
        # История обучения (сохраняется в модели)
        self.train_losses = []
        self.val_losses = []
        self.hyperparams = {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'img_size': config.IMG_SIZE,
            'epochs': config.EPOCHS
        }
        
        print(f"✅ Используется устройство: {self.device}")

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
            print('❌ ОШИБКА: Тренировочный датасет пуст!')
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

    def train(self, train_loader, val_loader=None, epochs=None):
        if train_loader is None or len(train_loader) == 0:
            print('❌ Невозможно обучить модель: нет данных')
            return
        if epochs is None:
            epochs = self.config.EPOCHS

        print(f"\n{'='*60}")
        print('🚀 НАЧАЛО ОБУЧЕНИЯ')
        print(f"{'='*60}")
        print(f"📊 Эпох: {epochs}")
        print(f"📦 Batch size: {self.config.BATCH_SIZE}")
        print(f"📈 Learning rate: {self.config.LEARNING_RATE}")
        print(f"🖼️  Размер изображений: {self.config.IMG_SIZE}")
        print(f"{'='*60}\n")

        self.model.train()
        self.train_losses = []
        self.val_losses = []
        
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
                    print(f'📚 Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                self.train_losses.append(avg_loss)
                
                # Валидация
                val_loss = None
                if val_loader is not None and len(val_loader) > 0:
                    val_loss = self._validate(val_loader)
                    if val_loss is not None:
                        self.val_losses.append(val_loss)
                
                # Вывод информации об эпохе
                print(f'\n✅ Эпоха {epoch + 1}/{epochs} завершена.')
                print(f'   📉 Train Loss: {avg_loss:.6f}')
                if val_loss is not None:
                    print(f'   📊 Val Loss:   {val_loss:.6f}')
                print(f'   ⚡ LR:         {self.optimizer.param_groups[0]["lr"]:.6f}')
                print('-' * 40)
        
        print(f"\n{'='*60}")
        print('🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО')
        print(f"{'='*60}")
        print(f"📉 Финальная train loss: {self.train_losses[-1]:.6f}")
        if self.val_losses:
            print(f"📊 Финальная val loss:   {self.val_losses[-1]:.6f}")
        print(f'📈 Улучшение: {self.train_losses[0] - self.train_losses[-1]:.6f}')
        print(f"{'='*60}\n")
        
        # Сохраняем модель и рисуем график
        self.save_model()
        if self.train_losses:
            plot_training_loss(self.train_losses, self.val_losses, self.config.OUTPUT_DIR)

    def _validate(self, val_loader):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else None

    def save_model(self, path=None):
        if path is None:
            path = self.config.MODEL_SAVE_PATH
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'hyperparams': self.hyperparams
        }, path)
        print(f'💾 Модель сохранена: {path}')
        return path

    def load_model(self, path=None):
        if path is None:
            path = self.config.MODEL_SAVE_PATH
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
            if 'hyperparams' in checkpoint:
                self.hyperparams = checkpoint['hyperparams']
            
            print(f'✅ Модель загружена: {path}')
            if self.train_losses:
                print(f'   📊 История обучения: {len(self.train_losses)} эпох')
                print(f'   📉 Финальная loss: {self.train_losses[-1]:.6f}')
            return True
        print(f'❌ Модель не найдена: {path}')
        return False

    def compute_metrics(self, original, processed):
        """Вычисление метрик качества изображения"""
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import mean_squared_error as mse
        
        metrics = {}
        
        # SSIM
        metrics['ssim'] = ssim(original, processed, data_range=1.0)
        
        # PSNR
        metrics['psnr'] = psnr(original, processed, data_range=1.0)
        
        # MSE
        metrics['mse'] = mse(original, processed)
        
        # LPIPS (упрощенная версия - среднеквадратичная разница)
        diff = np.abs(original - processed)
        metrics['lpips'] = np.mean(diff)
        
        # VIF (упрощенная версия - корреляция Пирсона)
        corr = np.corrcoef(original.flatten(), processed.flatten())[0, 1]
        metrics['vif'] = max(0, corr)
        
        # VMAF (упрощенная версия - комбинация SSIM и PSNR)
        metrics['vmaf'] = 50 * metrics['ssim'] + 0.2 * metrics['psnr']
        metrics['vmaf'] = min(100, max(0, metrics['vmaf']))
        
        return metrics

    def process_image(self, image_path, save_results=True):
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image.resize(self.config.IMG_SIZE), dtype=np.float32) / 255.0

            ordered = self.halftoner.ordered_dithering(image_np)
            error_diff = self.halftoner.error_diffusion(image_np)
            adaptive_result = self.halftoner.adaptive_halftoning(image_np, self.model, self.device, return_details=True)
            adaptive = adaptive_result['halftone']
            adaptive_preview = adaptive_result['preview']

            gray_image = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
            
            # Вычисление всех метрик
            metrics_ordered = self.compute_metrics(gray_image, ordered)
            metrics_error = self.compute_metrics(gray_image, error_diff)
            metrics_adaptive = self.compute_metrics(gray_image, adaptive)
            metrics_preview = self.compute_metrics(gray_image, adaptive_preview)

            if save_results:
                plot_results(
                    image_np, ordered, error_diff, adaptive, adaptive_preview,
                    metrics_ordered, metrics_error, metrics_adaptive, metrics_preview,
                    os.path.basename(image_path), self.config.OUTPUT_DIR
                )
                self._save_single_outputs(image_path, ordered, error_diff, adaptive, adaptive_preview)

            return {
                'ordered': metrics_ordered,
                'error_diffusion': metrics_error,
                'adaptive': metrics_adaptive,
                'adaptive_preview': metrics_preview,
                'image_name': os.path.basename(image_path)
            }
        except Exception as e:
            print(f'❌ Ошибка при обработке {image_path}: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _save_single_outputs(self, image_path, ordered, error_diff, adaptive, adaptive_preview):
        from PIL import Image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        Image.fromarray((ordered * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_ordered.png'))
        Image.fromarray((error_diff * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_error_diffusion.png'))
        Image.fromarray((adaptive * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_adaptive.png'))
        Image.fromarray((adaptive_preview * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_stable_preview.png'))