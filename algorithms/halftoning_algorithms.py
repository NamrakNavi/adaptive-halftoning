import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class HalftoningAlgorithms:
    def __init__(self, config):
        self.config = config

    def ordered_dithering(self, image):
        """
        Классическое упорядоченное растрирование (Bayer 4x4).
        Ожидает изображение в диапазоне [0, 1] или [0, 255].
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        h, w = image.shape
        output = np.zeros((h, w), dtype=np.float32)

        bayer_matrix = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ], dtype=np.float32) / 16.0

        bayer_size = 4
        for i in range(0, h, bayer_size):
            for j in range(0, w, bayer_size):
                for m in range(bayer_size):
                    for n in range(bayer_size):
                        if i + m < h and j + n < w:
                            threshold = bayer_matrix[m, n]
                            output[i + m, j + n] = 1.0 if image[i + m, j + n] > threshold else 0.0

        return output

    def error_diffusion(self, image):
        """
        Floyd–Steinberg error diffusion.
        Ожидает изображение в диапазоне [0, 1] или [0, 255].
        """
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)

        img_float = image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        h, w = img_float.shape
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

    def adaptive_halftoning(self, image, model, device, return_details=False):
        # 1. Приведение входа
        if isinstance(image, np.ndarray):
            image_np = image.astype(np.float32)
            if image_np.max() > 1.0:
                image_np = image_np / 255.0
        else:
            raise TypeError('adaptive_halftoning expects numpy.ndarray image')

        # 2. В градации серого
        if len(image_np.shape) == 3:
            gray_image = (
                0.299 * image_np[:, :, 0] +
                0.587 * image_np[:, :, 1] +
                0.114 * image_np[:, :, 2]
            )
        else:
            gray_image = image_np

        gray_image = np.clip(gray_image.astype(np.float32), 0.0, 1.0)
        h, w = gray_image.shape

        # 3. Приводим к CHW для модели
        if len(image_np.shape) == 2:
            image_chw = np.stack([image_np, image_np, image_np], axis=0)
        elif len(image_np.shape) == 3:
            if image_np.shape[0] == 3 and image_np.ndim == 3:
                image_chw = image_np
            else:
                image_chw = np.transpose(image_np, (2, 0, 1))
        else:
            raise ValueError(f'Unsupported image shape: {image_np.shape}')

        image_tensor = torch.FloatTensor(image_chw).unsqueeze(0).to(device)

        # 4. Прогон через U-Net → карта параметров
        model.eval()
        with torch.no_grad():
            param_map = model(image_tensor)
            param_map_np = param_map.squeeze().cpu().numpy()

        if len(param_map_np.shape) == 3:
            param_map_np = np.mean(param_map_np, axis=0)

        param_map_np = param_map_np.astype(np.float32)

        patch_size = self.config.PATCH_SIZE
        output = np.zeros((h, w), dtype=np.float32)
        threshold_map = np.zeros((h, w), dtype=np.float32)

        # 5. Патчевое адаптивное порогование
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                i_end = min(i + patch_size, h)
                j_end = min(j + patch_size, w)
                patch = gray_image[i:i_end, j:j_end]
                if patch.size == 0:
                    continue

                param = float(param_map_np[i:i_end, j:j_end].mean())
                mu = float(patch.mean())
                sigma = float(max(patch.std(), 0.01))
                threshold = mu + sigma * (2.0 * param - 1.0)

                threshold_map[i:i_end, j:j_end] = threshold
                patch_output = (patch > threshold).astype(np.float32)

                # Ограничим долю точек в патче
                black_ratio = float(patch_output.mean())
                min_ratio = 0.05
                max_ratio = 0.95

                if black_ratio < min_ratio:
                    adjust = np.percentile(patch, 100 * (1.0 - min_ratio))
                    patch_output = (patch > adjust).astype(np.float32)
                elif black_ratio > max_ratio:
                    adjust = np.percentile(patch, 100 * (1.0 - max_ratio))
                    patch_output = (patch > adjust).astype(np.float32)

                output[i:i_end, j:j_end] = patch_output

        from scipy.ndimage import uniform_filter

        
        # ==== Блок визуализации ====
        # Бинарная карта Adaptive
        halftone = output.astype(np.float32)

        # Плавная карта параметров → вероятностная карта в [0,1]
        # Нормализуем param_map_np, чтобы он вёл себя как prob-karte у Differentiable
        p = param_map_np.copy()

        # Линейная нормализация в [0,1], на случай, если модель выдаёт неплотно нормированные значения
        p_min, p_max = float(p.min()), float(p.max())
        if p_max > p_min:
            prob_map = (p - p_min) / (p_max - p_min)
        else:
            prob_map = np.zeros_like(p, dtype=np.float32)

        # Возможный мягкий blur для устранения артефактов карты параметров
        prob_map = gaussian_filter(prob_map, sigma=0.8)

        # Preview напрямую из вероятностной карты
        preview = np.clip(prob_map.astype(np.float32), 0.0, 1.0)

        if return_details:
            return {
                'halftone': halftone,
                'preview': preview,
                'param_map': param_map_np.astype(np.float32),
                'threshold_map': threshold_map.astype(np.float32),
                'gray': gray_image.astype(np.float32),
            }

        return halftone