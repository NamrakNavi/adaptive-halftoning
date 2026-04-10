import numpy as np
import torch

class HalftoningAlgorithms:
    #Реализация различных алгоритмов растрирования

    def __init__(self, config):
        self.config = config

    def ordered_dithering(self, image):
        #Упорядоченное растрирование
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)

        h, w = image.shape
        output = np.zeros((h, w))

        # Матрица Байера 4x4
        bayer_matrix = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ]) / 16.0

        bayer_size = 4

        for i in range(0, h, bayer_size):
            for j in range(0, w, bayer_size):
                for m in range(bayer_size):
                    for n in range(bayer_size):
                        if i+m < h and j+n < w:
                            threshold = bayer_matrix[m, n]
                            output[i+m, j+n] = 1 if image[i+m, j+n] > threshold else 0

        return output

    def error_diffusion(self, image):
        #Стохастическое растрирование (Floyd-Steinberg)
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)

        h, w = image.shape
        img_float = image.copy()

        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        output = np.zeros((h, w))

        for i in range(h):
            for j in range(w):
                old_pixel = img_float[i, j]
                new_pixel = 1 if old_pixel > 0.5 else 0
                output[i, j] = new_pixel
                quant_error = old_pixel - new_pixel

                if j + 1 < w:
                    img_float[i, j+1] += quant_error * 7/16
                if i + 1 < h and j - 1 >= 0:
                    img_float[i+1, j-1] += quant_error * 3/16
                if i + 1 < h:
                    img_float[i+1, j] += quant_error * 5/16
                if i + 1 < h and j + 1 < w:
                    img_float[i+1, j+1] += quant_error * 1/16

        return output

    def adaptive_halftoning(self, image, model, device):
        #Адаптивное растрирование с использованием CNN
        # Подготовка изображения
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=0)
            elif len(image.shape) == 3:
                if image.shape[0] != 3:
                    image = np.transpose(image, (2, 0, 1))

        # В тензор
        image_tensor = torch.FloatTensor(image).unsqueeze(0).to(device)

        # Получение карты параметров
        with torch.no_grad():
            param_map = model(image_tensor)

        param_map_np = param_map.squeeze().cpu().numpy()
        if len(param_map_np.shape) == 3:
            param_map_np = np.mean(param_map_np, axis=0)

        # Преобразование изображения
        if len(image.shape) == 3:
            gray_image = np.mean(image, axis=0)
        else:
            gray_image = image

        h, w = gray_image.shape
        output = np.zeros((h, w))

        # Обработка патчами
        patch_size = self.config.PATCH_SIZE
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                i_end = min(i + patch_size, h)
                j_end = min(j + patch_size, w)

                patch = gray_image[i:i_end, j:j_end]
                if patch.size == 0:
                    continue

                param = param_map_np[i:i_end, j:j_end].mean()
                mu = patch.mean()
                sigma = max(patch.std(), 0.01)

                # Адаптивный порог
                threshold = mu + sigma * (2 * param - 1)
                patch_output = (patch > threshold).astype(float)
                output[i:i_end, j:j_end] = patch_output

                    # Применяем Gaussian blur для сглаживания
        from scipy.ndimage import gaussian_filter
        output = gaussian_filter(output, sigma=0.5)
    
        # Или билатеральный фильтр (сохраняет границы)
        from scipy.ndimage import uniform_filter
        output = uniform_filter(output, size=3)
        return output