import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
        self.criterion = nn.MSELoss()
        self.halftoner = HalftoningAlgorithms(config)
        ensure_dir(config.OUTPUT_DIR)
        print(f'Используется устройство: {self.device}')
        print(f'Путь к датасету: {dataset_path}')
        print(f'Путь для результатов: {config.OUTPUT_DIR}')

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
            mode='train',
            color_type=train_color,
            category=train_category,
            transform=transform,
            target_size=self.config.IMG_SIZE
        )
        test_dataset = HalftoningDataset(
            self.config.DATASET_PATH,
            mode='test',
            color_type=test_color,
            category=test_category,
            transform=transform,
            target_size=self.config.IMG_SIZE
        )

        if len(train_dataset) == 0:
            print('ОШИБКА: Тренировочный датасет пуст!')
            return None, None

        batch_size = self._get_optimal_batch_size(len(train_dataset))
        print(f'Изображений для обучения: {len(train_dataset)}')
        print(f'Batch size: {batch_size}')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = None
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False, num_workers=2)
            print(f'Изображений для теста: {len(test_dataset)}')

        return train_loader, test_loader

    def _get_optimal_batch_size(self, dataset_size):
        if dataset_size < self.config.MIN_BATCH_SIZE:
            return max(1, dataset_size)
        elif dataset_size < self.config.BATCH_SIZE:
            return max(self.config.MIN_BATCH_SIZE, min(dataset_size, self.config.BATCH_SIZE))
        return self.config.BATCH_SIZE

    def train(self, train_loader, epochs=None):
        if train_loader is None or len(train_loader) == 0:
            print('Невозможно обучить модель: нет данных')
            return

        if epochs is None:
            epochs = self.config.EPOCHS

        self.model.train()
        print(f'Начало обучения на {len(train_loader.dataset)} изображениях...')
        print(f'Количество эпох: {epochs}')
        print(f'Батчей за эпоху: {len(train_loader)}')

        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                if images.size(0) == 0:
                    continue

                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                if torch.isnan(loss):
                    print(f'Пропуск batch {batch_idx}: loss = NaN')
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
            from PIL import Image
            from skimage.metrics import structural_similarity as ssim

            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image.resize(self.config.IMG_SIZE), dtype=np.float32) / 255.0

            ordered = self.halftoner.ordered_dithering(image_np)
            error_diff = self.halftoner.error_diffusion(image_np)
            dbs_screen = self.halftoner.dbs_screen_dithering(image_np)
            diff_halftone = self.halftoner.differentiable_halftoning(image_np)
            adaptive_data = self.halftoner.adaptive_halftoning(image_np, self.model, self.device, return_details=True)
            adaptive = adaptive_data['halftone']
            adaptive_preview = adaptive_data['preview']

            gray_image = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
            ssim_ordered = ssim(gray_image, ordered, data_range=1.0)
            ssim_error = ssim(gray_image, error_diff, data_range=1.0)
            ssim_dbs = ssim(gray_image, dbs_screen, data_range=1.0)
            ssim_diff = ssim(gray_image, diff_halftone, data_range=1.0)
            ssim_adaptive = ssim(gray_image, adaptive, data_range=1.0)
            ssim_preview = ssim(gray_image, adaptive_preview, data_range=1.0)

            if save_results:
                plot_results(
                    image_np, ordered, error_diff, dbs_screen, diff_halftone, adaptive, adaptive_preview,
                    ssim_ordered, ssim_error, ssim_dbs, ssim_diff, ssim_adaptive, ssim_preview,
                    os.path.basename(image_path), self.config.OUTPUT_DIR
                )

            return {
                'ssim_ordered': ssim_ordered,
                'ssim_error': ssim_error,
                'ssim_dbs': ssim_dbs,
                'ssim_diff': ssim_diff,
                'ssim_adaptive': ssim_adaptive,
                'ssim_preview': ssim_preview,
                'image_name': os.path.basename(image_path)
            }
        except Exception as e:
            print(f'Ошибка при обработке {image_path}: {e}')
            return None

    def batch_process(self, test_color='all', test_category='all', max_images=20):
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(self.config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dataset = HalftoningDataset(
            self.config.DATASET_PATH,
            mode='test',
            color_type=test_color,
            category=test_category,
            transform=transform,
            target_size=self.config.IMG_SIZE
        )

        if len(test_dataset) == 0:
            print('Нет тестовых изображений')
            return []

        print(f'Найдено {len(test_dataset)} изображений для обработки')
        results = []
        successful = 0

        for i in range(min(len(test_dataset), max_images)):
            img_path = test_dataset.image_paths[i]
            print(f'[{i + 1}/{min(len(test_dataset), max_images)}] Обработка: {os.path.basename(img_path)}')
            result = self.process_image(img_path, save_results=True)
            if result:
                results.append(result)
                successful += 1

        if results:
            self._print_summary_statistics(results, successful)
            self._save_summary_report(results)

        return results

    def _print_summary_statistics(self, results, successful):
        print('\n' + '=' * 60)
        print('ИТОГОВАЯ СТАТИСТИКА:')
        print('=' * 60)

        avg_ordered = np.mean([r['ssim_ordered'] for r in results])
        avg_error = np.mean([r['ssim_error'] for r in results])
        avg_dbs = np.mean([r['ssim_dbs'] for r in results])
        avg_diff = np.mean([r['ssim_diff'] for r in results])
        avg_adaptive = np.mean([r['ssim_adaptive'] for r in results])
        avg_preview = np.mean([r['ssim_preview'] for r in results])

        print(f'Успешно обработано: {successful}/{len(results)}')
        print(f'Средний SSIM (упорядоченное): {avg_ordered:.3f}')
        print(f'Средний SSIM (error diffusion): {avg_error:.3f}')
        print(f'Средний SSIM (DBS screen): {avg_dbs:.3f}')
        print(f'Средний SSIM (differentiable): {avg_diff:.3f}')
        print(f'Средний SSIM (адаптивное): {avg_adaptive:.3f}')
        print(f'Средний SSIM (preview): {avg_preview:.3f}')
        print('=' * 60)

    def _save_summary_report(self, results):
        import datetime
        report_path = os.path.join(
            self.config.OUTPUT_DIR,
            f"summary_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write('СВОДНЫЙ ОТЧЕТ ПО РАСТРИРОВАНИЮ\n')
            f.write('=' * 60 + '\n\n')
            f.write(f'Датасет: {self.config.DATASET_PATH}\n')
            f.write(f'Обработано изображений: {len(results)}\n\n')
            f.write('РЕЗУЛЬТАТЫ ПО ИЗОБРАЖЕНИЯМ:\n')
            f.write('-' * 60 + '\n')

            for result in results:
                f.write(f"\n{result['image_name']}:\n")
                f.write(f"  Упорядоченное: {result['ssim_ordered']:.3f}\n")
                f.write(f"  Error Diffusion: {result['ssim_error']:.3f}\n")
                f.write(f"  DBS Screen: {result['ssim_dbs']:.3f}\n")
                f.write(f"  Differentiable: {result['ssim_diff']:.3f}\n")
                f.write(f"  Адаптивное: {result['ssim_adaptive']:.3f}\n")
                f.write(f"  Preview: {result['ssim_preview']:.3f}\n")

        print(f'Сводный отчет сохранен: {report_path}')