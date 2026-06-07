import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_loss(losses, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'График обучения сохранен: {plot_path}')


def plot_results(original, ordered, error_diff, dbs_screen, differentiable, adaptive, adaptive_preview,
                 ssim_ordered, ssim_error, ssim_dbs, ssim_diff, ssim_adaptive, ssim_preview,
                 image_name, output_dir):
    fig, axes = plt.subplots(3, 4, figsize=(24, 14))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Оригинал')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ordered, cmap='gray')
    axes[0, 1].set_title(f'Упорядоченное\nSSIM: {ssim_ordered:.3f}')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(error_diff, cmap='gray')
    axes[0, 2].set_title(f'Error Diffusion\nSSIM: {ssim_error:.3f}')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(dbs_screen, cmap='gray')
    axes[0, 3].set_title(f'DBS Screen\nSSIM: {ssim_dbs:.3f}')
    axes[0, 3].axis('off')

    axes[1, 0].imshow(differentiable, cmap='gray')
    axes[1, 0].set_title(f'Differentiable\nSSIM: {ssim_diff:.3f}')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(adaptive, cmap='gray')
    axes[1, 1].set_title(f'Адаптивное\nSSIM: {ssim_adaptive:.3f}')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(adaptive_preview, cmap='gray')
    axes[1, 2].set_title(f'Карта порога / preview\nSSIM: {ssim_preview:.3f}')
    axes[1, 2].axis('off')

    axes[1, 3].axis('off')
    axes[1, 3].text(0.5, 0.5, 'Сравнение методов', ha='center', va='center', fontsize=14)
    axes[1, 3].axis('off')

    axes[2, 0].hist(original.flatten(), bins=50, alpha=0.7)
    axes[2, 0].set_title('Гистограмма оригинала')

    axes[2, 1].hist(ordered.flatten(), bins=50, alpha=0.7)
    axes[2, 1].set_title('Гистограмма упорядоченного')

    axes[2, 2].hist(error_diff.flatten(), bins=50, alpha=0.7)
    axes[2, 2].set_title('Гистограмма error diffusion')

    axes[2, 3].hist(adaptive.flatten(), bins=50, alpha=0.7)
    axes[2, 3].set_title('Гистограмма адаптивного')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Результаты сохранены: {save_path}')
    print(
        f'SSIM - Упорядоченное: {ssim_ordered:.3f}, '
        f'Error Diffusion: {ssim_error:.3f}, '
        f'DBS Screen: {ssim_dbs:.3f}, '
        f'Differentiable: {ssim_diff:.3f}, '
        f'Адаптивное: {ssim_adaptive:.3f}, Preview: {ssim_preview:.3f}')