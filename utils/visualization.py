import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_loss(losses, output_dir):
    """Визуализация кривой обучения"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График обучения сохранен: {plot_path}")

def plot_results(original, ordered, error_diff, adaptive,
                ssim_ordered, ssim_error, ssim_adaptive,
                image_name, output_dir):
    # Построение графиков результатов
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Оригинальное изображение
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Оригинал')
    axes[0, 0].axis('off')

    # Упорядоченное растрирование
    axes[0, 1].imshow(ordered, cmap='gray')
    axes[0, 1].set_title(f'Упорядоченное\nSSIM: {ssim_ordered:.3f}')
    axes[0, 1].axis('off')

    # Стохастическое растрирование
    axes[0, 2].imshow(error_diff, cmap='gray')
    axes[0, 2].set_title(f'Error Diffusion\nSSIM: {ssim_error:.3f}')
    axes[0, 2].axis('off')

    # Адаптивное растрирование
    axes[0, 3].imshow(adaptive, cmap='gray')
    axes[0, 3].set_title(f'Адаптивное (CNN)\nSSIM: {ssim_adaptive:.3f}')
    axes[0, 3].axis('off')

    # Гистограммы
    axes[1, 0].hist(original.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title('Гистограмма оригинала')

    axes[1, 1].hist(ordered.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Гистограмма упорядоченного')

    axes[1, 2].hist(error_diff.flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_title('Гистограмма error diffusion')

    axes[1, 3].hist(adaptive.flatten(), bins=50, alpha=0.7)
    axes[1, 3].set_title('Гистограмма адаптивного')

    plt.tight_layout()

    # Сохранение
    save_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Результаты сохранены: {save_path}")
    print(f"SSIM - Упорядоченное: {ssim_ordered:.3f}, Error Diffusion: {ssim_error:.3f}, Адаптивное: {ssim_adaptive:.3f}")