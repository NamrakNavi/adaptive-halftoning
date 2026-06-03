import os
import numpy as np
import torch
from torchvision import transforms


class HalftoningAlgorithms:
    """Реализация алгоритмов растрирования.
    Адаптивный метод реализован по матмодели:
    1) U-Net предсказывает карту параметров p_ij
    2) p_ij ограничивается в [0, 1] через sigmoid
    3) локальный порог T_ij = mu_ij + sigma_ij * (2 * p_ij - 1)
    4) бинаризация A_ij = 1, если I_ij > T_ij, иначе 0
    """

    def __init__(self, config):
        self.config = config

    def ordered_dithering(self, image):
        if len(image.shape) == 3:
            image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

        h, w = image.shape
        output = np.zeros((h, w), dtype=np.float32)
        bayer_matrix = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ], dtype=np.float32) / 16.0

        for i in range(h):
            for j in range(w):
                threshold = bayer_matrix[i % 4, j % 4]
                output[i, j] = 1.0 if image[i, j] > threshold else 0.0
        return output

    def error_diffusion(self, image):
        if len(image.shape) == 3:
            image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

        h, w = image.shape
        img_float = image.astype(np.float32).copy()
        output = np.zeros((h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                old_pixel = img_float[i, j]
                new_pixel = 1.0 if old_pixel > 0.5 else 0.0
                output[i, j] = new_pixel
                quant_error = old_pixel - new_pixel

                if j + 1 < w:
                    img_float[i, j + 1] += quant_error * 7 / 16
                if i + 1 < h and j - 1 >= 0:
                    img_float[i + 1, j - 1] += quant_error * 3 / 16
                if i + 1 < h:
                    img_float[i + 1, j] += quant_error * 5 / 16
                if i + 1 < h and j + 1 < w:
                    img_float[i + 1, j + 1] += quant_error * 1 / 16

        return np.clip(output, 0.0, 1.0)

    def _load_dbs_screen(self, screen_path=None):
        if screen_path and os.path.exists(screen_path):
            return np.loadtxt(screen_path, dtype=np.float32)

        default_path = getattr(self.config, 'DBS_SCREEN_PATH', None)
        if default_path and os.path.exists(default_path):
            return np.loadtxt(default_path, dtype=np.float32)

        try:
            from urllib.request import urlopen
            url = 'https://raw.githubusercontent.com/BaekduChoi/Halftoning_v2/main/screen_dbs_256.txt'
            with urlopen(url, timeout=10) as response:
                raw_text = response.read().decode('utf-8').strip().splitlines()
            screen = np.array(
                [[float(value) for value in line.split()] for line in raw_text],
                dtype=np.float32
            )
            if screen.size == 0:
                raise ValueError('Пустой DBS screen')
            return screen / np.max(screen)
        except Exception:
            return self._generate_dbs_screen(256)

    def _generate_dbs_screen(self, size):
        x = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)
        y = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        screen = (np.sin(2.0 * np.pi * (xx + yy * 2.0)) + 1.0) * 0.5
        return screen.astype(np.float32)

    def _resize_threshold_map(self, threshold_map, target_shape):
        if threshold_map.shape == target_shape:
            return threshold_map
        source_h, source_w = threshold_map.shape
        target_h, target_w = target_shape
        rows = np.minimum((np.arange(target_h) * source_h // target_h).astype(np.int32), source_h - 1)
        cols = np.minimum((np.arange(target_w) * source_w // target_w).astype(np.int32), source_w - 1)
        return threshold_map[rows[:, None], cols]

    def dbs_screen_dithering(self, image, screen_path=None):
        gray = self._to_gray(image)
        screen = self._load_dbs_screen(screen_path)
        if screen.ndim != 2:
            screen = np.squeeze(screen)
        screen = np.clip(screen, 0.0, 1.0)
        screen = self._resize_threshold_map(screen, gray.shape)
        return (gray > screen).astype(np.float32)

    def differentiable_halftoning(self, image, patch_size=None, scale=12.0):
        gray = self._to_gray(image)
        patch_size = patch_size or getattr(self.config, 'PATCH_SIZE', 8)
        mu_map, sigma_map = self._local_mean_std(gray, patch_size=patch_size)
        eps = 1e-6
        activations = (gray - mu_map) / (sigma_map + eps)
        soft = 1.0 / (1.0 + np.exp(-scale * activations))
        return (soft > 0.5).astype(np.float32)

    def _to_gray(self, image):
        if len(image.shape) == 3:
            return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float32)
        return image.astype(np.float32)

    def _local_mean_std(self, gray, patch_size=8):
        h, w = gray.shape
        mu_map = np.zeros((h, w), dtype=np.float32)
        sigma_map = np.zeros((h, w), dtype=np.float32)

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                i_end = min(i + patch_size, h)
                j_end = min(j + patch_size, w)
                patch = gray[i:i_end, j:j_end]
                if patch.size == 0:
                    continue
                mu = float(np.mean(patch))
                sigma = float(np.std(patch))
                sigma = max(sigma, 1e-3)
                mu_map[i:i_end, j:j_end] = mu
                sigma_map[i:i_end, j:j_end] = sigma
        return mu_map, sigma_map

    def adaptive_halftoning(self, image, model, device, return_details=False):
        gray = self._to_gray(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if image.ndim == 2:
            rgb = np.stack([image, image, image], axis=-1)
        else:
            rgb = image

        image_tensor = transform(rgb).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            raw_param_map = model(image_tensor)
            param_map = torch.sigmoid(raw_param_map)

        param_map_np = param_map.squeeze().detach().cpu().numpy().astype(np.float32)
        if param_map_np.ndim == 3:
            param_map_np = np.mean(param_map_np, axis=0)

        if param_map_np.shape != gray.shape:
            import cv2
            param_map_np = cv2.resize(param_map_np, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

        param_map_np = np.clip(param_map_np, 0.0, 1.0)
        patch_size = getattr(self.config, 'PATCH_SIZE', 8)
        mu_map, sigma_map = self._local_mean_std(gray, patch_size=patch_size)

        threshold_map = mu_map + sigma_map * (2.0 * param_map_np - 1.0)
        threshold_map = np.clip(threshold_map, 0.0, 1.0)
        adaptive = (gray > threshold_map).astype(np.float32)

        normal_preview = np.clip(threshold_map, 0.0, 1.0)

        if return_details:
            return {
                'halftone': adaptive,
                'preview': normal_preview,
                'param_map': param_map_np,
                'threshold_map': threshold_map,
                'gray': gray,
            }
        return adaptive