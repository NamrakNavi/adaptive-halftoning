import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

from models.unet import UNet
from algorithms.halftoning_algorithms import HalftoningAlgorithms
from data.dataset import HalftoningDataset
from utils.file_utils import ensure_dir
from utils.metrics import compute_quality_metrics, fid_metric
from utils.training_logger import TrainingLogger
from utils.visualization import plot_results, plot_training_loss


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
        ensure_dir(config.LOGS_DIR)
        ensure_dir(config.PLOTS_DIR)
        ensure_dir(config.METRICS_DIR)
        print(f'Используется устройство: {self.device}')

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

        logger = TrainingLogger(self.config.OUTPUT_DIR, self.config)
        self.model.train()
        losses = []
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0.0
            num_batches = 0

            for batch_idx, (images, targets) in enumerate(train_loader, start=1):
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

                loss_value = loss.item()
                total_loss += loss_value
                num_batches += 1
                lr = self.optimizer.param_groups[0]['lr']
                logger.log_batch(epoch + 1, batch_idx, len(train_loader), loss_value, lr)

                if batch_idx % max(1, len(train_loader) // 5) == 0:
                    print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss_value:.6f}, LR: {lr:.8f}')

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                epoch_duration = time.time() - epoch_start
                losses.append(avg_loss)
                logger.log_epoch(epoch + 1, avg_loss, self.optimizer.param_groups[0]['lr'], epoch_duration, num_batches)
                print(f'Эпоха {epoch + 1} завершена. Средняя loss: {avg_loss:.6f}. Время: {epoch_duration:.2f} сек.')

        self.save_model()
        logger.finalize()
        if losses:
            plot_training_loss(losses, self.config.PLOTS_DIR)

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
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image.resize(self.config.IMG_SIZE), dtype=np.float32) / 255.0
            gray_image = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]

            ordered = self.halftoner.ordered_dithering(image_np)
            error_diff = self.halftoner.error_diffusion(image_np)
            adaptive_result = self.halftoner.adaptive_halftoning(image_np, self.model, self.device, return_details=True)
            adaptive = adaptive_result['halftone']
            adaptive_preview = adaptive_result['preview']
            threshold_map = adaptive_result['threshold_map']
            param_map = adaptive_result['param_map']

            metrics_by_method = {
                'Ordered': compute_quality_metrics(gray_image, ordered, include_optional=self.config.ENABLE_OPTIONAL_METRICS),
                'Error Diffusion': compute_quality_metrics(gray_image, error_diff, include_optional=self.config.ENABLE_OPTIONAL_METRICS),
                'Adaptive': compute_quality_metrics(gray_image, adaptive, include_optional=self.config.ENABLE_OPTIONAL_METRICS),
                'Adaptive Preview': compute_quality_metrics(gray_image, adaptive_preview, include_optional=self.config.ENABLE_OPTIONAL_METRICS),
            }

            if save_results:
                plot_results(
                    image_np, ordered, error_diff, adaptive, adaptive_preview,
                    threshold_map, param_map, metrics_by_method,
                    os.path.basename(image_path), self.config.OUTPUT_DIR
                )
                self._save_single_outputs(image_path, ordered, error_diff, adaptive, adaptive_preview, threshold_map, param_map)
                self._save_metrics_csv(os.path.basename(image_path), metrics_by_method)

            return {
                'image_name': os.path.basename(image_path),
                'metrics': metrics_by_method,
                'ssim_ordered': metrics_by_method['Ordered']['SSIM'],
                'ssim_error': metrics_by_method['Error Diffusion']['SSIM'],
                'ssim_adaptive': metrics_by_method['Adaptive']['SSIM'],
                'ssim_adaptive_preview': metrics_by_method['Adaptive Preview']['SSIM'],
            }
        except Exception as e:
            print(f'Ошибка при обработке {image_path}: {e}')
            return None

    def _save_single_outputs(self, image_path, ordered, error_diff, adaptive, adaptive_preview, threshold_map, param_map):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        Image.fromarray((ordered * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_ordered.png'))
        Image.fromarray((error_diff * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_error_diffusion.png'))
        Image.fromarray((adaptive * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_adaptive.png'))
        Image.fromarray((adaptive_preview * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_adaptive_preview.png'))
        Image.fromarray((threshold_map * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_threshold_map.png'))
        Image.fromarray((param_map * 255).astype(np.uint8)).save(os.path.join(self.config.OUTPUT_DIR, f'{base_name}_param_map.png'))

    def _save_metrics_csv(self, image_name, metrics_by_method):
        csv_path = os.path.join(self.config.METRICS_DIR, f'{os.path.splitext(image_name)[0]}_metrics.csv')
        keys = set()
        for method_metrics in metrics_by_method.values():
            keys.update(method_metrics.keys())
        keys = ['method'] + sorted(keys)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for method_name, metric_values in metrics_by_method.items():
                row = {'method': method_name}
                row.update(metric_values)
                writer.writerow(row)
        print(f'Метрики сохранены: {csv_path}')

    def batch_process(self, test_color='all', test_category='all', max_images=20):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(self.config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dataset = HalftoningDataset(
            self.config.DATASET_PATH,
            mode='test', color_type=test_color, category=test_category,
            transform=transform, target_size=self.config.IMG_SIZE
        )
        if len(test_dataset) == 0:
            print('Нет тестовых изображений')
            return []

        results = []
        gray_refs = []
        adaptive_preds = []

        for i in range(min(len(test_dataset), max_images)):
            img_path = test_dataset.image_paths[i]
            result = self.process_image(img_path, save_results=True)
            if result:
                results.append(result)
                image = Image.open(img_path).convert('RGB')
                image_np = np.array(image.resize(self.config.IMG_SIZE), dtype=np.float32) / 255.0
                gray_refs.append(0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2])
                adaptive_preds.append(self.halftoner.adaptive_halftoning(image_np, self.model, self.device))

        if results:
            fid_value = fid_metric(gray_refs, adaptive_preds)
            self._save_summary_report(results, fid_value)
            self._print_summary_statistics(results, fid_value)

        return results

    def _print_summary_statistics(self, results, fid_value=None):
        adaptive_ssim = [r['metrics']['Adaptive']['SSIM'] for r in results]
        adaptive_psnr = [r['metrics']['Adaptive']['PSNR'] for r in results]
        adaptive_mse = [r['metrics']['Adaptive']['MSE'] for r in results]
        adaptive_iou = [r['metrics']['Adaptive']['IoU'] for r in results]
        print('\n' + '=' * 60)
        print('ИТОГОВАЯ СТАТИСТИКА')
        print('=' * 60)
        print(f'Обработано изображений: {len(results)}')
        print(f'Средний SSIM (Adaptive): {np.mean(adaptive_ssim):.4f}')
        print(f'Средний PSNR (Adaptive): {np.mean(adaptive_psnr):.4f}')
        print(f'Средний MSE (Adaptive): {np.mean(adaptive_mse):.6f}')
        print(f'Средний IoU (Adaptive): {np.mean(adaptive_iou):.4f}')
        if fid_value is not None:
            print(f'FID-like score по батчу: {fid_value:.6f}')
        print('=' * 60)

    def _save_summary_report(self, results, fid_value=None):
        report_path = os.path.join(self.config.OUTPUT_DIR, 'summary_report_metrics.csv')
        fieldnames = ['image_name', 'method', 'SSIM', 'PSNR', 'MSE', 'IoU', 'LPIPS', 'BRISQUE', 'NIQE']
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                for method_name, metric_values in result['metrics'].items():
                    row = {'image_name': result['image_name'], 'method': method_name}
                    row.update(metric_values)
                    writer.writerow(row)
        if fid_value is not None:
            fid_path = os.path.join(self.config.METRICS_DIR, 'batch_fid.txt')
            with open(fid_path, 'w', encoding='utf-8') as f:
                f.write(f'FID-like score: {fid_value}\n')
        print(f'Сводный отчет сохранен: {report_path}')
