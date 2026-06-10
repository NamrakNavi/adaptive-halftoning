import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from DISTS_pytorch import DISTS          

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
        print(f'Используется устройство: {self.device}')

        self.dists_metric = DISTS().to(self.device).eval()              # DISTS

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

    def train(self, train_loader, val_loader=None, epochs=None):
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
            for batch in train_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                    continue
                images, targets = batch
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

                total_loss += float(loss.item())
                num_batches += 1

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                losses.append(avg_loss)
                print(f'Epoch {epoch + 1}/{epochs}, loss = {avg_loss:.6f}')

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

    def _to_gray(self, image_np):
        if image_np.ndim == 3:
            return (0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]).astype(np.float32)
        return image_np.astype(np.float32)
    
    def _to_tensor_01(self, img_np: np.ndarray) -> torch.Tensor:
        img = np.asarray(img_np, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)  # gray → псевдо-RGB

        img_chw = np.transpose(img, (2, 0, 1))       # HWC → CHW
        tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        return tensor

    def _calculate_metrics(self, reference, candidate):
        reference = np.clip(reference.astype(np.float32), 0.0, 1.0)
        candidate = np.clip(candidate.astype(np.float32), 0.0, 1.0)

        # БАЗОВЫЕ МЕТРИКИ (как было)
        mse = float(np.mean((reference - candidate) ** 2))
        mae = float(np.mean(np.abs(reference - candidate)))
        rmse = float(np.sqrt(mse))
        psnr = float('inf') if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))
        ssim_value = float(ssim(reference, candidate, data_range=1.0))

        metrics = {
            'SSIM': ssim_value,
            'PSNR': psnr,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
        }


        try:
            ref_t = self._to_tensor_01(reference)
            cand_t = self._to_tensor_01(candidate)

            with torch.no_grad():
                dists_val = float(self.dists_metric(ref_t, cand_t).item())
                metrics['DISTS'] = dists_val
                
        except Exception as e:
            print(f'Предупреждение: не удалось посчитать DISTS/IQT: {e}')

        return metrics

    def _save_single_outputs(self, image_path, result):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for method_key, payload in result.items():
            if method_key in ('image_name', 'original', 'original_gray'):
                continue
            if not isinstance(payload, dict) or 'image' not in payload:
                continue
            image = np.clip(payload['image'], 0.0, 1.0)
            save_name = payload.get('save_name', method_key)
            Image.fromarray((image * 255).astype(np.uint8)).save(
                os.path.join(self.config.OUTPUT_DIR, f'{base_name}_{save_name}.png')
            )

    def _differentiable(self, image_np):
        """Differentiable-метод: полутоновая карта + бинаризация с адаптивным порогом."""
        if isinstance(image_np, np.ndarray):
            img = image_np.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
        else:
            raise TypeError('Differentiable ожидает numpy.ndarray')

        # Приводим к CHW
        if img.ndim == 2:
            img_chw = np.stack([img, img, img], axis=0)
        elif img.ndim == 3:
            if img.shape[0] == 3:
                img_chw = img
            else:
                img_chw = np.transpose(img, (2, 0, 1))
        else:
            raise ValueError(f'Неподдерживаемая форма изображения: {img.shape}')

        tensor = torch.FloatTensor(img_chw).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)          # [1, 1, H, W]
            prob = torch.sigmoid(logits).cpu().numpy().squeeze().astype(np.float32)

        # Нормальное (полутоновое) представление
        preview = np.clip(prob, 0.0, 1.0)

        # Адаптивный порог: медиана + сдвиг (чтобы избежать полностью черного/белого)
        p_med = float(np.median(preview))
        p_std = float(preview.std())
        thresh = p_med + 0.1 * p_std  # можно крутить 0.1

        binary = (preview > thresh).astype(np.float32)

        return {
            'binary': binary,
            'preview': preview,
        }

    def process_image(self, image_path, save_results=True):
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image.resize(self.config.IMG_SIZE), dtype=np.float32) / 255.0
            gray_image = self._to_gray(image_np)

            ordered = self.halftoner.ordered_dithering(image_np)
            error_diff = self.halftoner.error_diffusion(image_np)
            adaptive_result = self.halftoner.adaptive_halftoning(image_np, self.model, self.device, return_details=True)
            adaptive = adaptive_result['halftone']
            adaptive_preview = adaptive_result['preview']
            differentiable_result = self._differentiable(image_np)
            differentiable_bin = differentiable_result['binary']
            differentiable_prev = differentiable_result['preview']

            result = {
                'image_name': os.path.basename(image_path),
                'original': image_np,
                'original_gray': gray_image,
                'ordered': {
                    'title': 'Ordered Dithering',
                    'save_name': 'ordered',
                    'image': ordered,
                    'metrics': self._calculate_metrics(gray_image, ordered)
                },
                'error_diffusion': {
                    'title': 'Error Diffusion',
                    'save_name': 'error_diffusion',
                    'image': error_diff,
                    'metrics': self._calculate_metrics(gray_image, error_diff)
                },
                'adaptive': {
                    'title': 'Adaptive Halftoning',
                    'save_name': 'adaptive',
                    'image': adaptive,
                    'metrics': self._calculate_metrics(gray_image, adaptive)
                },
                'differentiable': {
                    'title': 'Differentiable',
                    'save_name': 'differentiable',
                    'image': np.clip(differentiable_bin, 0.0, 1.0),
                    'metrics': self._calculate_metrics(gray_image, np.clip(differentiable_bin, 0.0, 1.0))
                },
                'differentiable_preview': {
                    'title': 'Differentiable Preview',
                    'save_name': 'differentiable_preview',
                    'image': np.clip(differentiable_prev, 0.0, 1.0),
                    'metrics': self._calculate_metrics(gray_image, np.clip(differentiable_prev, 0.0, 1.0))
                },
                'adaptive_preview': {
                    'title': 'Adaptive Preview',
                    'save_name': 'adaptive_preview',
                    'image': np.clip(adaptive_preview, 0.0, 1.0),
                    'metrics': self._calculate_metrics(gray_image, np.clip(adaptive_preview, 0.0, 1.0))
                }
            }

            if save_results:
                plot_results(image_np, result, os.path.basename(image_path), self.config.OUTPUT_DIR)
                self._save_single_outputs(image_path, result)

            return result
        except Exception as e:
            print(f'Ошибка при обработке {image_path}: {e}')
            return None

    def batch_process(self, test_color='all', test_category='all', max_images=20):
        results = []
        test_dir = os.path.join(self.config.DATASET_PATH, 'test')
        if not os.path.exists(test_dir):
            print(f'Тестовая папка не найдена: {test_dir}')
            return results

        image_extensions = tuple(self.config.IMAGE_EXTENSIONS)
        image_paths = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))

        if test_color != 'all':
            image_paths = [p for p in image_paths if test_color in p]
        if test_category != 'all':
            image_paths = [p for p in image_paths if test_category in p]

        for image_path in image_paths[:max_images]:
            result = self.process_image(image_path, save_results=True)
            if result:
                results.append(result)

        return results