import csv
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt


class TrainingLogger:
    def __init__(self, output_dir, config):
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, 'logs')
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_prefix = f'training_{timestamp}'
        self.batch_csv_path = os.path.join(self.logs_dir, f'{self.session_prefix}_batch_log.csv')
        self.epoch_csv_path = os.path.join(self.logs_dir, f'{self.session_prefix}_epoch_log.csv')
        self.hyperparams_json_path = os.path.join(self.logs_dir, f'{self.session_prefix}_hyperparameters.json')

        self.batch_rows = []
        self.epoch_rows = []
        self.start_time = time.time()
        self._save_hyperparameters(config)

    def _save_hyperparameters(self, config):
        serializable = {}
        for key, value in config.__dict__.items():
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                serializable[key] = value
            else:
                serializable[key] = str(value)
        with open(self.hyperparams_json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def log_batch(self, epoch, batch_idx, total_batches, loss, lr):
        row = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_batches': total_batches,
            'loss': float(loss),
            'lr': float(lr),
            'elapsed_sec': float(time.time() - self.start_time),
        }
        self.batch_rows.append(row)

    def log_epoch(self, epoch, avg_loss, lr, epoch_duration, train_batches):
        row = {
            'epoch': epoch,
            'avg_loss': float(avg_loss),
            'lr': float(lr),
            'epoch_duration_sec': float(epoch_duration),
            'train_batches': int(train_batches),
            'elapsed_sec': float(time.time() - self.start_time),
        }
        self.epoch_rows.append(row)

    def finalize(self):
        self._write_csv(self.batch_csv_path, self.batch_rows)
        self._write_csv(self.epoch_csv_path, self.epoch_rows)
        self._plot_training_progress()
        return {
            'batch_csv': self.batch_csv_path,
            'epoch_csv': self.epoch_csv_path,
            'hyperparams_json': self.hyperparams_json_path,
        }

    def _write_csv(self, path, rows):
        if not rows:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _plot_training_progress(self):
        if not self.epoch_rows:
            return

        epochs = [r['epoch'] for r in self.epoch_rows]
        losses = [r['avg_loss'] for r in self.epoch_rows]
        lrs = [r['lr'] for r in self.epoch_rows]
        durations = [r['epoch_duration_sec'] for r in self.epoch_rows]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs, losses, marker='o', linewidth=2)
        axes[0].set_title('Кривая обучения')
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Средняя функция потерь')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, lrs, marker='s', linewidth=2, color='tab:orange')
        axes[1].set_title('Изменение learning rate')
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Learning rate')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, durations, marker='^', linewidth=2, color='tab:green')
        axes[2].set_title('Время эпох')
        axes[2].set_xlabel('Эпоха')
        axes[2].set_ylabel('Время, сек')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'{self.session_prefix}_training_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Графики прогресса обучения сохранены: {save_path}')
