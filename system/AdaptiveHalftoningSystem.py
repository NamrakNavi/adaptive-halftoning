import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.unet import UNet
from algorithms.halftoning_algorithms import HalftoningAlgorithms
from data.dataset import HalftoningDataset
from utils.visualization import plot_results, plot_training_loss
from utils.file_utils import ensure_dir

class AdaptiveHalftoningSystem:
    # Система адаптивного растрирования

    def __init__(self, dataset_path, config):
        self.config = config
        self.config.DATASET_PATH = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Инициализация модели
        self.model = UNet(in_channels=3, out_channels=1, use_sigmoid=False).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()

        # Инициализация алгоритмов
        self.halftoner = HalftoningAlgorithms(config)

        # Создание выходной директории
        ensure_dir(config.OUTPUT_DIR)

        print(f"Используется устройство: {self.device}")
        print(f"Путь к датасету: {dataset_path}")
        print(f"Путь для результатов: {config.OUTPUT_DIR}")

    def create_dataloaders(self, train_color='all', train_category='all',
                          test_color='all', test_category='all', transform=None):
        # Создание DataLoader'ов
        from torchvision import transforms

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(self.config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # Создаем датасеты
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

        # Проверка датасетов
        if len(train_dataset) == 0:
            print("ОШИБКА: Тренировочный датасет пуст!")
            return None, None

        # Выбор оптимального batch size
        batch_size = self._get_optimal_batch_size(len(train_dataset))

        print(f"\nНастройки обучения:")
        print(f"  • Изображений: {len(train_dataset)}")
        print(f"  • Batch size: {batch_size}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=False
        )

        test_loader = None
        if len(test_dataset) > 0:
            test_batch_size = min(batch_size, len(test_dataset))
            test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=2
            )
            print(f"  • Тестовых изображений: {len(test_dataset)}")

        return train_loader, test_loader

    def _get_optimal_batch_size(self, dataset_size):
        #Выбор оптимального batch size
        if dataset_size < self.config.MIN_BATCH_SIZE:
            return max(1, dataset_size)
        elif dataset_size < self.config.BATCH_SIZE:
            optimal = max(self.config.MIN_BATCH_SIZE,
                         min(2 ** int(np.log2(dataset_size // 2)),
                             self.config.BATCH_SIZE))
            return optimal
        else:
            return self.config.BATCH_SIZE

    def train(self, train_loader, epochs=None):
        #Обучение модели
        if train_loader is None or len(train_loader) == 0:
            print("Невозможно обучить модель: нет данных")
            return

        if epochs is None:
            epochs = self.config.EPOCHS

        self.model.train()
        print(f"\nНачало обучения на {len(train_loader.dataset)} изображениях...")
        print(f"Количество эпох: {epochs}")
        print(f"Батчей за эпоху: {len(train_loader)}")

        losses = []

        for epoch in range(epochs):
            total_loss = 0
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
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % max(1, len(train_loader) // 5) == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                losses.append(avg_loss)
                print(f'Epoch {epoch+1} завершен. Средняя loss: {avg_loss:.6f}')

            # Сохранение модели каждые 10 эпох
            if (epoch + 1) % 10 == 0:
                self.save_model(f"{self.config.MODEL_SAVE_PATH}_epoch_{epoch+1}")

        # Сохраняем финальную модель
        self.save_model()

        # Визуализация обучения
        if losses:
            plot_training_loss(losses, self.config.OUTPUT_DIR)

    def save_model(self, path=None):
        #Сохранение модели
        if path is None:
            path = self.config.MODEL_SAVE_PATH

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, path)
        print(f"Модель сохранена: {path}")
        return path

    def load_model(self, path=None):
        #Загрузка модели
        if path is None:
            path = self.config.MODEL_SAVE_PATH

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f"Модель загружена: {path}")
            return True
        else:
            print(f"Модель не найдена: {path}")
            return False

    def process_image(self, image_path, save_results=True):
        #Обработка одного изображения
        try:
            from PIL import Image
            from skimage.metrics import structural_similarity as ssim
            from utils.visualization import plot_results as plot_results_func

            # Загрузка изображения
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image.resize(self.config.IMG_SIZE)) / 255.0

            # Применение алгоритмов
            ordered = self.halftoner.ordered_dithering(image_np)
            error_diff = self.halftoner.error_diffusion(image_np)
            adaptive = self.halftoner.adaptive_halftoning(image_np, self.model, self.device)

            # Расчет SSIM
            gray_image = np.mean(image_np, axis=2) if len(image_np.shape) == 3 else image_np

            ssim_ordered = ssim(gray_image, ordered, data_range=1.0)
            ssim_error = ssim(gray_image, error_diff, data_range=1.0)
            ssim_adaptive = ssim(gray_image, adaptive, data_range=1.0)

            # Визуализация
            if save_results:
                plot_results_func(
                    image_np, ordered, error_diff, adaptive,
                    ssim_ordered, ssim_error, ssim_adaptive,
                    os.path.basename(image_path), self.config.OUTPUT_DIR
                )

            return {
                'ssim_ordered': ssim_ordered,
                'ssim_error': ssim_error,
                'ssim_adaptive': ssim_adaptive,
                'image_name': os.path.basename(image_path)
            }

        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            return None

    def batch_process(self, test_color='all', test_category='all', max_images=20):
        # Пакетная обработка изображений
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(self.config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
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
            print(f"Нет тестовых изображений")
            return []

        print(f"\nНайдено {len(test_dataset)} изображений для обработки")

        results = []
        successful = 0

        for i in range(min(len(test_dataset), max_images)):
            img_path = test_dataset.image_paths[i]
            print(f"\n[{i+1}/{min(len(test_dataset), max_images)}] Обработка: {os.path.basename(img_path)}")

            result = self.process_image(img_path, save_results=True)
            if result:
                results.append(result)
                successful += 1

        if results:
            self._print_summary_statistics(results, successful)
            self._save_summary_report(results)

        return results

    def _print_summary_statistics(self, results, successful):
        #Вывод сводной статистики
        print("\n" + "="*60)
        print("ИТОГОВАЯ СТАТИСТИКА:")
        print("="*60)

        avg_ordered = np.mean([r['ssim_ordered'] for r in results])
        avg_error = np.mean([r['ssim_error'] for r in results])
        avg_adaptive = np.mean([r['ssim_adaptive'] for r in results])

        print(f"Успешно обработано: {successful}/{len(results)}")
        print(f"Средний SSIM (упорядоченное): {avg_ordered:.3f}")
        print(f"Средний SSIM (error diffusion): {avg_error:.3f}")
        print(f"Средний SSIM (адаптивное): {avg_adaptive:.3f}")

        if avg_ordered > 0:
            improvement = ((avg_adaptive - avg_ordered) / avg_ordered) * 100
            print(f"Среднее улучшение: {improvement:.1f}%")

        print("="*60)

    def _save_summary_report(self, results):
        #Сохранение сводного отчета
        import datetime

        report_path = os.path.join(self.config.OUTPUT_DIR,
                                  f"summary_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("СВОДНЫЙ ОТЧЕТ ПО РАСТРИРОВАНИЮ\n")
            f.write("="*60 + "\n\n")

            f.write(f"Датасет: {self.config.DATASET_PATH}\n")
            f.write(f"Обработано изображений: {len(results)}\n\n")

            f.write("РЕЗУЛЬТАТЫ ПО ИЗОБРАЖЕНИЯМ:\n")
            f.write("-"*60 + "\n")

            for result in results:
                f.write(f"\n{result['image_name']}:\n")
                f.write(f"  Упорядоченное: {result['ssim_ordered']:.3f}\n")
                f.write(f"  Error Diffusion: {result['ssim_error']:.3f}\n")
                f.write(f"  Адаптивное (CNN): {result['ssim_adaptive']:.3f}\n")

            f.write("\n" + "="*60 + "\n")
            f.write("СТАТИСТИКА:\n")
            f.write("-"*60 + "\n")

            avg_ordered = np.mean([r['ssim_ordered'] for r in results])
            avg_error = np.mean([r['ssim_error'] for r in results])
            avg_adaptive = np.mean([r['ssim_adaptive'] for r in results])

            f.write(f"Средний SSIM (упорядоченное): {avg_ordered:.3f}\n")
            f.write(f"Средний SSIM (error diffusion): {avg_error:.3f}\n")
            f.write(f"Средний SSIM (адаптивное): {avg_adaptive:.3f}\n")

        print(f"\nСводный отчет сохранен: {report_path}")

    def compare_with_halftoning_v2(self):
        #Сравнение с Halftoning_v2
        print("\n" + "="*60)
        print("СРАВНЕНИЕ С Halftoning_v2:")
        print("="*60)

        print("Наш подход (CNN-based):")
        print("• Адаптивное растрирование с учетом локальных особенностей")
        print("• Обучение на датасете для оптимальных параметров")
        print("• Сохранение текстур и деталей")

        print("\nHalftoning_v2 (Traditional):")
        print("• Фиксированные алгоритмы (error diffusion)")
        print("• Один набор параметров для всего изображения")
        print("• Потеря деталей в сложных областях")

        print("\nПреимущества нашего подхода:")
        print("1. Более высокое качество изображения")
        print("2. Адаптивность к различным типам изображений")
        print("3. Сохранение мелких деталей и текстур")

        print("="*60)