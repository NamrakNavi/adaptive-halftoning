import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_loss(train_losses, val_losses=None, output_dir=None):
    """Визуализация кривой обучения"""
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    
    if val_losses and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if output_dir:
        plot_path = os.path.join(output_dir, 'training_loss.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 График обучения сохранен: {plot_path}")
    
    plt.close()


def plot_results(original, ordered, error_diff, adaptive, adaptive_preview,
                metrics_ordered, metrics_error, metrics_adaptive, metrics_preview,
                image_name, output_dir):
    """Построение графиков результатов (без гистограмм)"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    
    # Верхний ряд - изображения
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Оригинал', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ordered, cmap='gray')
    axes[0, 1].set_title(f'Упорядоченное\nSSIM: {metrics_ordered["ssim"]:.4f}', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(error_diff, cmap='gray')
    axes[0, 2].set_title(f'Error Diffusion\nSSIM: {metrics_error["ssim"]:.4f}', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(adaptive, cmap='gray')
    axes[0, 3].set_title(f'Адаптивное (CNN)\nSSIM: {metrics_adaptive["ssim"]:.4f}', fontsize=10)
    axes[0, 3].axis('off')
    
    # Нижний ряд - сравнение метрик
    metrics_names = ['SSIM', 'PSNR/50', '1-LPIPS', 'VIF', 'VMAF/100']
    methods = ['Ordered', 'Error Diff', 'Adaptive', 'Preview']
    
    data = {
        'Ordered': [
            metrics_ordered['ssim'],
            metrics_ordered['psnr'] / 50,
            1 - metrics_ordered['lpips'],
            metrics_ordered['vif'],
            metrics_ordered['vmaf'] / 100
        ],
        'Error Diff': [
            metrics_error['ssim'],
            metrics_error['psnr'] / 50,
            1 - metrics_error['lpips'],
            metrics_error['vif'],
            metrics_error['vmaf'] / 100
        ],
        'Adaptive': [
            metrics_adaptive['ssim'],
            metrics_adaptive['psnr'] / 50,
            1 - metrics_adaptive['lpips'],
            metrics_adaptive['vif'],
            metrics_adaptive['vmaf'] / 100
        ],
        'Preview': [
            metrics_preview['ssim'],
            metrics_preview['psnr'] / 50,
            1 - metrics_preview['lpips'],
            metrics_preview['vif'],
            metrics_preview['vmaf'] / 100
        ]
    }
    
    x = np.arange(len(metrics_names))
    width = 0.2
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#73AB84']
    
    for i, (method, values) in enumerate(data.items()):
        offset = (i - 1.5) * width
        bars = axes[1, 0].bar(x + offset, values, width, label=method, color=colors[i], alpha=0.8)
        for bar, val in zip(bars, values):
            if val > 0.05:
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    axes[1, 0].set_xlabel('Метрики', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Нормализованное значение', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Сравнение метрик качества', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics_names)
    axes[1, 0].legend(loc='upper right', fontsize=9)
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Убираем неиспользуемые подграфики
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')
    
    fig.suptitle(f'Результаты растрирования: {os.path.splitext(image_name)[0]}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{os.path.splitext(image_name)[0]}_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Результаты сохранены: {save_path}")
    print(f"\n{'='*60}")
    print(f"📊 МЕТРИКИ ДЛЯ {image_name}")
    print(f"{'='*60}")
    print(f"\n{'Метрика':<12} {'Ordered':<12} {'Error Diff':<12} {'Adaptive':<12}")
    print(f"{'-'*50}")
    print(f"SSIM       {metrics_ordered['ssim']:<12.4f} {metrics_error['ssim']:<12.4f} {metrics_adaptive['ssim']:<12.4f}")
    print(f"PSNR (dB)  {metrics_ordered['psnr']:<12.2f} {metrics_error['psnr']:<12.2f} {metrics_adaptive['psnr']:<12.2f}")
    print(f"MSE        {metrics_ordered['mse']:<12.6f} {metrics_error['mse']:<12.6f} {metrics_adaptive['mse']:<12.6f}")
    print(f"LPIPS      {metrics_ordered['lpips']:<12.4f} {metrics_error['lpips']:<12.4f} {metrics_adaptive['lpips']:<12.4f}")
    print(f"VIF        {metrics_ordered['vif']:<12.4f} {metrics_error['vif']:<12.4f} {metrics_adaptive['vif']:<12.4f}")
    print(f"VMAF       {metrics_ordered['vmaf']:<12.2f} {metrics_error['vmaf']:<12.2f} {metrics_adaptive['vmaf']:<12.2f}")
    print(f"{'='*60}\n")