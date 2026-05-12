import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter


class HalftoningAlgorithms:
    def __init__(self, config):
        self.config = config

    def _to_gray(self, image):
        if len(image.shape) == 3:
            return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return image.astype(np.float32)

    def ordered_dithering(self, image):
        image = self._to_gray(image)
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
                output[i, j] = 1.0 if image[i, j] > bayer_matrix[i % 4, j % 4] else 0.0
        return output

    def error_diffusion(self, image):
        gray = self._to_gray(image)
        h, w = gray.shape
        img_float = gray.copy().astype(np.float32)
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
        gray = self._to_gray(image).astype(np.float32)
        gray = np.clip(gray, 0.0, 1.0)

        if len(image.shape) == 2:
            image_chw = np.stack([gray, gray, gray], axis=0)
        else:
            image_chw = np.transpose(image.astype(np.float32), (2, 0, 1))

        image_tensor = torch.FloatTensor(image_chw).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(image_tensor)
            pred_map = torch.sigmoid(pred).squeeze().detach().cpu().numpy().astype(np.float32)

        if pred_map.ndim == 3:
            pred_map = pred_map.mean(axis=0)

        pred_map = gaussian_filter(pred_map, sigma=self.config.GAUSSIAN_SIGMA)
        pred_map = np.clip(pred_map, 0.0, 1.0)

        local_mean = uniform_filter(gray, size=self.config.LOCAL_WINDOW, mode='reflect')
        local_sq_mean = uniform_filter(gray ** 2, size=self.config.LOCAL_WINDOW, mode='reflect')
        local_std = np.sqrt(np.clip(local_sq_mean - local_mean ** 2, 0.0, 1.0))

        base_threshold = 0.5 + (pred_map - 0.5) * self.config.ADAPTIVE_STRENGTH
        detail_term = (gray - local_mean) * 0.25
        contrast_term = (local_std - local_std.mean()) * 0.35
        adaptive_threshold = base_threshold - detail_term - contrast_term
        adaptive_threshold = np.clip(
            adaptive_threshold,
            self.config.MIN_THRESHOLD,
            self.config.MAX_THRESHOLD
        )

        smooth_gray = gaussian_filter(gray, sigma=self.config.GAUSSIAN_SIGMA)
        raw_adaptive = (smooth_gray > adaptive_threshold).astype(np.float32)

        error_base = self.error_diffusion(np.dstack([gray, gray, gray]))
        mixed = (1.0 - self.config.MIX_WITH_ERROR_DIFFUSION) * raw_adaptive +                 self.config.MIX_WITH_ERROR_DIFFUSION * error_base
        halftone = (mixed >= 0.5).astype(np.float32)

        if self.config.ENABLE_MEDIAN_POST:
            halftone = median_filter(halftone, size=3)
            halftone = (halftone >= 0.5).astype(np.float32)

        stability_preview = gaussian_filter(halftone, sigma=1.2)
        stability_preview = np.clip(0.82 * smooth_gray + 0.18 * stability_preview, 0.0, 1.0)

        if return_details:
            return {
                'halftone': halftone.astype(np.float32),
                'preview': stability_preview.astype(np.float32),
                'param_map': pred_map.astype(np.float32),
                'threshold_map': adaptive_threshold.astype(np.float32),
                'gray': gray.astype(np.float32)
            }

        return halftone.astype(np.float32)
