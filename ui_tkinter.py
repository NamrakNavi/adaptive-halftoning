import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from system.AdaptiveHalftoningSystem import AdaptiveHalftoningSystem
from utils.file_utils import (
    find_dataset_windows,
    normalize_windows_path,
    create_dataset_structure,
    check_dataset_structure,
)


class TkTextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        if not text:
            return
        self.widget.after(0, self._append, text)

    def flush(self):
        pass

    def _append(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)


class AdaptiveHalftoningUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Система адаптивного растрирования')
        self.root.geometry('1100x760')
        self.root.minsize(980, 680)

        self.config = Config()
        self.system = None
        self.current_model_path = self.config.MODEL_SAVE_PATH
        self.current_image_path = ''

        self._build_styles()
        self._build_ui()
        self._init_defaults()
        self._redirect_stdout()

    def _build_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(main, text='Пути и общие параметры', padding=10)
        top.pack(fill=tk.X, pady=(0, 10))

        self.dataset_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.image_var = tk.StringVar()
        self.epochs_var = tk.StringVar()
        self.batch_var = tk.StringVar()
        self.lr_var = tk.StringVar()
        self.img_w_var = tk.StringVar()
        self.img_h_var = tk.StringVar()
        self.max_images_var = tk.StringVar(value='20')

        self.train_color_var = tk.StringVar(value='all')
        self.train_category_var = tk.StringVar(value='all')
        self.test_color_var = tk.StringVar(value='all')
        self.test_category_var = tk.StringVar(value='all')

        self._path_row(top, 0, 'Датасет:', self.dataset_var, self.choose_dataset)
        self._path_row(top, 1, 'Результаты:', self.output_var, self.choose_output_dir)
        self._path_row(top, 2, 'Модель:', self.model_var, self.choose_model_file)
        self._path_row(top, 3, 'Изображение:', self.image_var, self.choose_image)

        params = ttk.Frame(top)
        params.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        for i in range(8):
            params.columnconfigure(i, weight=1)

        ttk.Label(params, text='Epochs').grid(row=0, column=0, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.epochs_var, width=10).grid(row=0, column=1, sticky='ew', padx=4, pady=4)
        ttk.Label(params, text='Batch Size').grid(row=0, column=2, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.batch_var, width=10).grid(row=0, column=3, sticky='ew', padx=4, pady=4)
        ttk.Label(params, text='Learning Rate').grid(row=0, column=4, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.lr_var, width=12).grid(row=0, column=5, sticky='ew', padx=4, pady=4)
        ttk.Label(params, text='Макс. изображений').grid(row=0, column=6, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.max_images_var, width=10).grid(row=0, column=7, sticky='ew', padx=4, pady=4)

        ttk.Label(params, text='Ширина').grid(row=1, column=0, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.img_w_var, width=10).grid(row=1, column=1, sticky='ew', padx=4, pady=4)
        ttk.Label(params, text='Высота').grid(row=1, column=2, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.img_h_var, width=10).grid(row=1, column=3, sticky='ew', padx=4, pady=4)

        filters = ttk.LabelFrame(main, text='Фильтры датасета', padding=10)
        filters.pack(fill=tk.X, pady=(0, 10))
        for i in range(8):
            filters.columnconfigure(i, weight=1)

        color_values = ['all', 'colored', 'black and white']
        category_values = ['all', 'nature', 'transport']

        ttk.Label(filters, text='Train color').grid(row=0, column=0, sticky='w', padx=4, pady=4)
        ttk.Combobox(filters, textvariable=self.train_color_var, values=color_values, state='readonly').grid(row=0, column=1, sticky='ew', padx=4, pady=4)
        ttk.Label(filters, text='Train category').grid(row=0, column=2, sticky='w', padx=4, pady=4)
        ttk.Combobox(filters, textvariable=self.train_category_var, values=category_values, state='readonly').grid(row=0, column=3, sticky='ew', padx=4, pady=4)
        ttk.Label(filters, text='Test color').grid(row=0, column=4, sticky='w', padx=4, pady=4)
        ttk.Combobox(filters, textvariable=self.test_color_var, values=color_values, state='readonly').grid(row=0, column=5, sticky='ew', padx=4, pady=4)
        ttk.Label(filters, text='Test category').grid(row=0, column=6, sticky='w', padx=4, pady=4)
        ttk.Combobox(filters, textvariable=self.test_category_var, values=category_values, state='readonly').grid(row=0, column=7, sticky='ew', padx=4, pady=4)

        actions = ttk.LabelFrame(main, text='Действия', padding=10)
        actions.pack(fill=tk.X, pady=(0, 10))

        btns = [
            ('Найти датасет', self.auto_find_dataset),
            ('Проверить датасет', self.check_dataset),
            ('Создать структуру датасета', self.create_dataset),
            ('Инициализировать систему', self.init_system),
            ('Обучить модель', self.train_model),
            ('Загрузить модель', self.load_model),
            ('Обработать изображение', self.process_image),
            ('Пакетная обработка', self.batch_process),
            ('Показать историю обучения', self.show_training_history),
            ('Очистить лог', self.clear_log),
        ]

        for idx, (text, cmd) in enumerate(btns):
            ttk.Button(actions, text=text, command=cmd).grid(row=idx // 3, column=idx % 3, sticky='ew', padx=6, pady=6)
        for i in range(3):
            actions.columnconfigure(i, weight=1)

        log_frame = ttk.LabelFrame(main, text='Лог работы', padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_widget = tk.Text(log_frame, wrap='word', height=22)
        self.log_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_widget.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_widget.configure(yscrollcommand=scroll.set)

        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(8, 0))
        self.status_var = tk.StringVar(value='Готово')
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

    def _path_row(self, parent, row, label, var, command):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=4, pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky='ew', padx=4, pady=4)
        ttk.Button(parent, text='Обзор', command=command).grid(row=row, column=2, sticky='ew', padx=4, pady=4)

    def _init_defaults(self):
        self.dataset_var.set(self.config.DATASET_PATH)
        self.output_var.set(self.config.OUTPUT_DIR)
        self.model_var.set(self.config.MODEL_SAVE_PATH)
        self.epochs_var.set(str(self.config.EPOCHS))
        self.batch_var.set(str(self.config.BATCH_SIZE))
        self.lr_var.set(str(self.config.LEARNING_RATE))
        self.img_w_var.set(str(self.config.IMG_SIZE[0]))
        self.img_h_var.set(str(self.config.IMG_SIZE[1]))

    def _redirect_stdout(self):
        redirector = TkTextRedirector(self.log_widget)
        sys.stdout = redirector
        sys.stderr = redirector

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_widget.delete('1.0', tk.END)

    def choose_dataset(self):
        path = filedialog.askdirectory(title='Выберите папку датасета')
        if path:
            self.dataset_var.set(normalize_windows_path(path))

    def choose_output_dir(self):
        path = filedialog.askdirectory(title='Выберите папку результатов')
        if path:
            self.output_var.set(normalize_windows_path(path))

    def choose_model_file(self):
        path = filedialog.askopenfilename(
            title='Выберите файл модели',
            filetypes=[('PyTorch model', '*.pth'), ('All files', '*.*')]
        )
        if path:
            self.model_var.set(normalize_windows_path(path))

    def choose_image(self):
        path = filedialog.askopenfilename(
            title='Выберите изображение',
            filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff'), ('All files', '*.*')]
        )
        if path:
            self.image_var.set(normalize_windows_path(path))

    def sync_config_from_ui(self):
        self.config.DATASET_PATH = normalize_windows_path(self.dataset_var.get().strip())
        self.config.OUTPUT_DIR = normalize_windows_path(self.output_var.get().strip())
        self.config.MODEL_SAVE_PATH = normalize_windows_path(self.model_var.get().strip())
        self.config.MODELS_DIR = os.path.dirname(self.config.MODEL_SAVE_PATH)
        self.config.EPOCHS = int(self.epochs_var.get().strip())
        self.config.BATCH_SIZE = int(self.batch_var.get().strip())
        self.config.LEARNING_RATE = float(self.lr_var.get().strip())
        self.config.IMG_SIZE = (int(self.img_w_var.get().strip()), int(self.img_h_var.get().strip()))
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.MODELS_DIR, exist_ok=True)

    def ensure_system(self):
        self.sync_config_from_ui()
        if self.system is None:
            self.system = AdaptiveHalftoningSystem(self.config.DATASET_PATH, self.config)
        return self.system

    def run_in_thread(self, target):
        thread = threading.Thread(target=self._safe_run, args=(target,), daemon=True)
        thread.start()

    def _safe_run(self, target):
        try:
            target()
            self.set_status('Готово')
        except Exception as e:
            self.set_status('Ошибка')
            traceback.print_exc()
            messagebox.showerror('Ошибка', str(e))

    def auto_find_dataset(self):
        def task():
            self.set_status('Поиск датасета...')
            path = find_dataset_windows(self.config)
            if path:
                self.dataset_var.set(normalize_windows_path(path))
                print(f'Найден датасет: {path}')
            else:
                print('Датасет не найден автоматически.')
        self.run_in_thread(task)

    def check_dataset(self):
        def task():
            self.set_status('Проверка датасета...')
            self.sync_config_from_ui()
            dataset_path = self.config.DATASET_PATH
            if not os.path.exists(dataset_path):
                print(f'Путь не существует: {dataset_path}')
                return
            structure_ok, stats = check_dataset_structure(dataset_path)
            print(f'Структура датасета корректна: {structure_ok}')
            print('Статистика:')
            print(stats)
        self.run_in_thread(task)

    def create_dataset(self):
        def task():
            self.set_status('Создание структуры датасета...')
            self.sync_config_from_ui()
            ok = create_dataset_structure(self.config.DATASET_PATH)
            if ok:
                print(f'Структура датасета создана: {self.config.DATASET_PATH}')
        self.run_in_thread(task)

    def init_system(self):
        def task():
            self.set_status('Инициализация системы...')
            self.system = None
            self.ensure_system()
            print('Система инициализирована.')
        self.run_in_thread(task)

    def train_model(self):
        def task():
            self.set_status('Обучение модели...')
            system = self.ensure_system()
            train_loader, test_loader = system.create_dataloaders(
                train_color=self.train_color_var.get(),
                train_category=self.train_category_var.get(),
                test_color=self.test_color_var.get(),
                test_category=self.test_category_var.get(),
            )
            if train_loader is None or len(train_loader) == 0:
                print('Не удалось создать train_loader. Обучение отменено.')
                return
            system.train(train_loader, val_loader=test_loader, epochs=self.config.EPOCHS)
            system.save_model(self.config.MODEL_SAVE_PATH)
            self.model_var.set(self.config.MODEL_SAVE_PATH)
            print(f'Обучение завершено. Модель сохранена: {self.config.MODEL_SAVE_PATH}')
        self.run_in_thread(task)

    def load_model(self):
        def task():
            self.set_status('Загрузка модели...')
            system = self.ensure_system()
            model_path = normalize_windows_path(self.model_var.get().strip())
            if not model_path or not os.path.exists(model_path):
                print(f'Файл модели не найден: {model_path}')
                return
            ok = system.load_model(model_path)
            if ok:
                print('Модель успешно загружена.')
        self.run_in_thread(task)

    def process_image(self):
        # Открываем диалог выбора файла
        image_path = filedialog.askopenfilename(
            title='Выберите изображение для обработки',
            filetypes=[
                ('Image files', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'),
                ('PNG files', '*.png'),
                ('JPEG files', '*.jpg *.jpeg'),
                ('All files', '*.*')
            ]
        )
        
        if not image_path:
            print("Обработка отменена: файл не выбран")
            return
        
        self.image_var.set(normalize_windows_path(image_path))
        
        def task():
            self.set_status('Обработка изображения...')
            system = self.ensure_system()
            
            if not os.path.exists(image_path):
                print(f'Файл изображения не найден: {image_path}')
                return
            
            model_path = normalize_windows_path(self.model_var.get().strip())
            if model_path and os.path.exists(model_path):
                system.load_model(model_path)
            
            result = system.process_image(image_path, save_results=True)
            if result:
                print(f'\n✅ Обработка завершена: {os.path.basename(image_path)}')
                print(f'   Упорядоченное:    SSIM={result["ordered"]["ssim"]:.4f}, PSNR={result["ordered"]["psnr"]:.2f}dB')
                print(f'   Error Diffusion: SSIM={result["error_diffusion"]["ssim"]:.4f}, PSNR={result["error_diffusion"]["psnr"]:.2f}dB')
                print(f'   Адаптивное:      SSIM={result["adaptive"]["ssim"]:.4f}, PSNR={result["adaptive"]["psnr"]:.2f}dB')
                
                messagebox.showinfo(
                    "Готово", 
                    f"Изображение обработано!\n\n"
                    f"Адаптивное растрирование:\n"
                    f"  SSIM = {result['adaptive']['ssim']:.4f}\n"
                    f"  PSNR = {result['adaptive']['psnr']:.2f} dB\n\n"
                    f"Результаты сохранены в:\n{self.config.OUTPUT_DIR}"
                )
        
        self.run_in_thread(task)

    def batch_process(self):
        def task():
            self.set_status('Пакетная обработка...')
            
            self.sync_config_from_ui()
            dataset_path = self.config.DATASET_PATH
            
            if not os.path.exists(dataset_path):
                print(f'❌ Ошибка: Путь к датасету не существует: {dataset_path}')
                return
            
            system = self.ensure_system()
            
            model_path = normalize_windows_path(self.model_var.get().strip())
            if model_path and os.path.exists(model_path):
                system.load_model(model_path)
                print(f'Загружена модель: {model_path}')
            
            max_images = int(self.max_images_var.get().strip())
            test_color = self.test_color_var.get()
            test_category = self.test_category_var.get()
            
            print(f'\n{"="*60}')
            print('ПАКЕТНАЯ ОБРАБОТКА')
            print(f'{"="*60}')
            print(f'Датасет: {dataset_path}')
            print(f'Цвет: {test_color}, Категория: {test_category}')
            print(f'Максимум изображений: {max_images}')
            print(f'{"="*60}\n')
            
            # Создаем датасет
            from data.dataset import HalftoningDataset
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize(self.config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            try:
                test_dataset = HalftoningDataset(
                    dataset_path,
                    mode='test',
                    color_type=test_color,
                    category=test_category,
                    transform=transform,
                    target_size=self.config.IMG_SIZE
                )
            except Exception as e:
                print(f'❌ Ошибка при создании датасета: {e}')
                return
            
            if len(test_dataset) == 0:
                print('❌ Ошибка: В датасете нет изображений для обработки!')
                return
            
            num_images = min(len(test_dataset), max_images) if max_images > 0 else len(test_dataset)
            print(f'Найдено изображений: {len(test_dataset)}')
            print(f'Будет обработано: {num_images}\n')
            
            results = []
            success = 0
            failed = 0
            
            for i in range(num_images):
                try:
                    img_path = test_dataset.image_paths[i]
                    print(f'[{i+1}/{num_images}] Обработка: {os.path.basename(img_path)}')
                    
                    result = system.process_image(img_path, save_results=True)
                    if result:
                        results.append(result)
                        success += 1
                        print(f'  ✓ SSIM: {result["adaptive"]["ssim"]:.4f}, PSNR: {result["adaptive"]["psnr"]:.2f}dB')
                    else:
                        failed += 1
                        print(f'  ✗ Ошибка обработки')
                except Exception as e:
                    failed += 1
                    print(f'  ✗ Ошибка: {e}')
            
            # Вывод итогов
            print(f'\n{"="*60}')
            print('ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ')
            print(f'{"="*60}')
            print(f'✅ Успешно: {success}')
            print(f'❌ Ошибок: {failed}')
            
            if results:
                avg_ssim_ordered = sum(r["ordered"]["ssim"] for r in results) / len(results)
                avg_ssim_adaptive = sum(r["adaptive"]["ssim"] for r in results) / len(results)
                avg_psnr_ordered = sum(r["ordered"]["psnr"] for r in results) / len(results)
                avg_psnr_adaptive = sum(r["adaptive"]["psnr"] for r in results) / len(results)
                
                print(f'\n📊 СРЕДНИЕ МЕТРИКИ (по {len(results)} изображениям):')
                print(f'   Упорядоченное:    SSIM = {avg_ssim_ordered:.4f}, PSNR = {avg_psnr_ordered:.2f} dB')
                print(f'   Адаптивное:       SSIM = {avg_ssim_adaptive:.4f}, PSNR = {avg_psnr_adaptive:.2f} dB')
            
            print(f'\n💾 Результаты сохранены в: {self.config.OUTPUT_DIR}')
            print(f'{"="*60}\n')
            
            messagebox.showinfo(
                "Пакетная обработка завершена",
                f"✅ Обработано: {success}\n❌ Ошибок: {failed}\n\n"
                f"📊 Средний SSIM (adaptive): {avg_ssim_adaptive:.4f}\n"
                f"📊 Средний PSNR (adaptive): {avg_psnr_adaptive:.2f} dB"
            )
        
        self.run_in_thread(task)

    def show_training_history(self):
        """Показать историю обучения из загруженной модели"""
        if self.system is None:
            print("❌ Система не инициализирована. Нажмите 'Инициализировать систему' или 'Загрузить модель'")
            return
        
        if not self.system.train_losses:
            print("📊 История обучения не найдена. Модель не обучена или загружена без истории.")
            return
        
        print(f'\n{"="*60}')
        print('ИСТОРИЯ ОБУЧЕНИЯ')
        print(f'{"="*60}')
        print(f'Количество эпох: {len(self.system.train_losses)}')
        print(f'Начальная loss: {self.system.train_losses[0]:.6f}')
        print(f'Финальная loss: {self.system.train_losses[-1]:.6f}')
        print(f'Улучшение: {(self.system.train_losses[0] - self.system.train_losses[-1]):.6f}')
        
        if self.system.val_losses:
            print(f'\nВалидация:')
            print(f'  Начальная val loss: {self.system.val_losses[0]:.6f}')
            print(f'  Финальная val loss: {self.system.val_losses[-1]:.6f}')
        
        print(f'\n{"-"*40}')
        print('Последние 10 эпох:')
        print(f'{"Эпоха":<8} {"Train Loss":<12} {"Val Loss":<12}')
        print(f'{"-"*40}')
        
        start = max(0, len(self.system.train_losses) - 10)
        for i in range(start, len(self.system.train_losses)):
            val_str = f"{self.system.val_losses[i]:.6f}" if i < len(self.system.val_losses) else "N/A"
            print(f'{i+1:<8} {self.system.train_losses[i]:<12.6f} {val_str}')
        
        print(f'{"="*60}\n')


def main():
    root = tk.Tk()
    app = AdaptiveHalftoningUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()