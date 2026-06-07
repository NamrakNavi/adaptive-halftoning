import math
import os
import matplotlib.pyplot as plt


def _format_metrics(metrics):
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            value_str = 'inf' if math.isinf(value) else f'{value:.4f}'
        else:
            value_str = str(value)
        lines.append(f'{key}: {value_str}')
    return '\n'.join(lines)


def plot_results(original, methods, image_name, output_dir):
    display_items = [
        ('original', {'title': 'Original', 'image': original, 'metrics': {}}),
    ]

    ordered_keys = [
        'ordered',
        'error_diffusion',
        'adaptive',
        'adaptive_preview',
        'differentiable',
        'differentiable_preview'
    ]

    for key in ordered_keys:
        if key in methods:
            display_items.append((key, methods[key]))

    n = len(display_items)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.4 * rows))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    flat_axes = [ax for row in axes for ax in row]

    for ax, (_, item) in zip(flat_axes, display_items):
        image = item['image']
        if getattr(image, 'ndim', 2) == 2:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)

        title = item.get('title', 'Метод')
        metrics = item.get('metrics', {})
        if metrics:
            ax.set_title(f'{title}\n{_format_metrics(metrics)}', fontsize=9)
        else:
            ax.set_title(title, fontsize=10)
        ax.axis('off')

    for ax in flat_axes[len(display_items):]:
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Результаты сохранены: {save_path}')


def plot_training_loss(losses, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.title('График функции потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(output_dir, 'training_loss.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'График loss сохранен: {save_path}')