import os

import matplotlib.pyplot as plt


def _format_metrics(metrics):
    lines = []
    for key, value in metrics.items():
        if value is None:
            lines.append(f'{key}: n/a')
        elif value == float('inf'):
            lines.append(f'{key}: inf')
        else:
            lines.append(f'{key}: {value:.4f}')
    return '\n'.join(lines)


def plot_training_loss(losses, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Средняя функция потерь')
    plt.title('Кривая обучения')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'График обучения сохранен: {plot_path}')


def plot_results(original, ordered, error_diff, adaptive, adaptive_preview,
                 threshold_map, param_map, metrics_by_method,
                 image_name, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Оригинал')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ordered, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Упорядоченное')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(error_diff, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Error Diffusion')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(adaptive, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title('Адаптивное')
    axes[0, 3].axis('off')

    axes[1, 0].imshow(adaptive_preview, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Adaptive Preview')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(threshold_map, cmap='viridis')
    axes[1, 1].set_title('Карта порога')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(param_map, cmap='magma')
    axes[1, 2].set_title('Карта параметров')
    axes[1, 2].axis('off')

    axes[1, 3].axis('off')
    text_blocks = []
    for method_name, metric_values in metrics_by_method.items():
        text_blocks.append(f'{method_name}\n' + _format_metrics(metric_values))
    axes[1, 3].text(0.01, 0.99, '\n\n'.join(text_blocks), va='top', ha='left', fontsize=10, family='monospace')
    axes[1, 3].set_title('Метрики')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Результаты сохранены: {save_path}')
