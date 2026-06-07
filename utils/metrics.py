import math
import warnings

import numpy as np
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings('ignore')


def _to_float01(image):
    image = np.asarray(image).astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _to_gray(image):
    image = _to_float01(image)
    if image.ndim == 2:
        return image
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float32)


def mse_metric(reference, candidate):
    ref = _to_gray(reference)
    cand = _to_gray(candidate)
    return float(np.mean((ref - cand) ** 2))


def psnr_metric(reference, candidate):
    mse = mse_metric(reference, candidate)
    if mse <= 1e-12:
        return float('inf')
    return float(10.0 * math.log10(1.0 / mse))


def ssim_metric(reference, candidate):
    ref = _to_gray(reference)
    cand = _to_gray(candidate)
    return float(ssim(ref, cand, data_range=1.0))


def iou_metric(reference, candidate, threshold=0.5):
    ref = (_to_gray(reference) >= threshold).astype(np.uint8)
    cand = (_to_gray(candidate) >= threshold).astype(np.uint8)
    intersection = np.logical_and(ref, cand).sum()
    union = np.logical_or(ref, cand).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def lpips_metric(reference, candidate):
    try:
        import torch
        import lpips

        loss_fn = lpips.LPIPS(net='alex')
        ref = _to_float01(reference)
        cand = _to_float01(candidate)
        if ref.ndim == 2:
            ref = np.stack([ref, ref, ref], axis=-1)
        if cand.ndim == 2:
            cand = np.stack([cand, cand, cand], axis=-1)
        ref_t = torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1
        cand_t = torch.from_numpy(cand.transpose(2, 0, 1)).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            score = loss_fn(ref_t, cand_t)
        return float(score.item())
    except Exception:
        return None


def brisque_metric(image):
    try:
        import cv2
        from imquality import brisque
        img = (_to_float01(image) * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return float(brisque.score(img))
    except Exception:
        return None


def niqe_metric(image):
    try:
        import cv2
        from skvideo.measure import niqe
        img = (_to_float01(image) * 255).astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return float(niqe(img))
    except Exception:
        return None


def fid_metric(reference_images, candidate_images):
    try:
        refs = np.stack([_to_gray(img).flatten() for img in reference_images], axis=0)
        cands = np.stack([_to_gray(img).flatten() for img in candidate_images], axis=0)
        mu1 = refs.mean(axis=0)
        mu2 = cands.mean(axis=0)
        diff = mu1 - mu2
        return float(np.mean(diff ** 2))
    except Exception:
        return None


def compute_quality_metrics(reference, candidate, include_optional=True):
    metrics = {
        'SSIM': ssim_metric(reference, candidate),
        'PSNR': psnr_metric(reference, candidate),
        'MSE': mse_metric(reference, candidate),
        'IoU': iou_metric(reference, candidate),
    }
    if include_optional:
        metrics['LPIPS'] = lpips_metric(reference, candidate)
        metrics['BRISQUE'] = brisque_metric(candidate)
        metrics['NIQE'] = niqe_metric(candidate)
    return metrics
