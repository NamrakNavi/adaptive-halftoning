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
    normalize_windows_path,
    find_dataset_windows,
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
            ('Копировать лог', self.copy_log),
            ('Сохранить лог', self.save_log),
            ('Очистить лог', self.clear_log),
        ]

        for idx, (text, cmd) in enumerate(btns):
            ttk.Button(actions, text=text, command=cmd).grid(
                row=idx // 4,
                column=idx % 4,
                sticky='ew',
                padx=6,
                pady=6
            )

        for i in range(4):
            actions.columnconfigure(i, weight=1)

        log_frame = ttk.LabelFrame(main, text='Лог работы', padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_widget = tk.Text(log_frame, wrap='word', height=22)
        self.log_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_widget.bind("<Control-c>", self._copy_selection)
        self.log_widget.bind("<Control-C>", self._copy_selection)
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

    def copy_log(self):
        try:
            text = self.log_widget.get('1.0', tk.END).strip()
            if not text:
                messagebox.showinfo('Лог', 'Лог пуст, копировать нечего.')
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
            print('Лог скопирован в буфер обмена.')
        except Exception as e:
            messagebox.showerror('Ошибка', f'Не удалось скопировать лог: {e}')


    def save_log(self):
        try:
            text = self.log_widget.get('1.0', tk.END).strip()
            if not text:
                messagebox.showinfo('Лог', 'Лог пуст, сохранять нечего.')
                return

            file_path = filedialog.asksaveasfilename(
                title='Сохранить лог',
                defaultextension='.txt',
                filetypes=[
                    ('Text files', '*.txt'),
                    ('Log files', '*.log'),
                    ('All files', '*.*')
                ]
            )

            if not file_path:
                return

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f'Лог сохранён: {file_path}')
        except Exception as e:
            messagebox.showerror('Ошибка', f'Не удалось сохранить лог: {e}')

    def _copy_selection(self, event=None):
        try:
            selected = self.log_widget.get("sel.first", "sel.last")
        except tk.TclError:
            # Ничего не выделено — просто выходим
            return "break"

        # Чистим буфер обмена и кладём туда текст
        self.root.clipboard_clear()
        self.root.clipboard_append(selected)
        return "break"

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
            filetypes=[
                ('Images', '*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff'),
                ('All files', '*.*')
            ]
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

    def _print_result_metrics(self, result):
        print(f'Результаты обработки: {result["image_name"]}')
        for method_key, payload in result.items():
            if method_key in ('image_name', 'original', 'original_gray'):
                continue
            if not isinstance(payload, dict):
                continue

            title = payload.get('title', method_key)
            metrics = payload.get('metrics', {})
            print(f'{title}:')

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    if metric_value == float('inf'):
                        value_str = 'inf'
                    else:
                        value_str = f'{metric_value:.6f}'
                else:
                    value_str = str(metric_value)
                print(f'  {metric_name} = {value_str}')

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
            def task():
                from tkinter import filedialog
                import os   # оставить

                self.set_status("Обработка изображения...")

                # 1. Каждый раз спрашиваем новый файл
                path = filedialog.askopenfilename(
                    title="Выберите изображение",
                    filetypes=(
                        ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                        ("All files", "*.*"),
                    ),
                )
                if not path:
                    print("Обработка отменена пользователем")
                    self.set_status("Готово")
                    return

                # 2. Сохраняем путь в поле и нормализуем
                self.image_var.set(path)
                image_path = normalize_windows_path(path)

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    self.set_status("Ошибка: файл не найден")
                    return

                # 3. Готовим систему и модель
                system = self.ensure_system()

                model_path = normalize_windows_path(self.model_var.get().strip())
                if model_path and os.path.exists(model_path):
                    system.load_model(model_path)

                # 4. Обрабатываем выбранное изображение
                result = system.process_image(image_path, save_results=True)
                if result:
                    print("\n[DEBUG] Структура metrics для Ordered:")
                    print(result.get('ordered', {}).get('metrics'))
                    print("\nРезультаты метрик:")
                    print(f"Файл: {result.get('image_name', '')}")

                    def show_metrics(name):
                        block = result.get(name, {})
                        title = block.get('title', name)
                        metrics = block.get('metrics', {})
                        print(f"\n{title}:")
                        print(f"  SSIM : {metrics.get('SSIM', 0.0):0.4f}")
                        print(f"  MSE  : {metrics.get('MSE', 0.0):0.4f}")
                        print(f"  PSNR : {metrics.get('PSNR', 0.0):0.4f} dB")
                        print(f"  MAE  : {metrics.get('MAE', 0.0):0.4f}")
                        print(f"  RMSE  : {metrics.get('RMSE', 0.0):0.4f}")

                    # Классические методы
                    show_metrics('ordered')
                    show_metrics('error_diffusion')

                    # CNN‑метод и его превью
                    show_metrics('adaptive')
                    show_metrics('adaptive_preview')

                    # Дифференцируемый метод и превью
                    show_metrics('differentiable')
                    show_metrics('differentiable_preview')

                self.set_status("Готово")

            self.run_in_thread(task)

    def batch_process(self):
        def task():
            self.set_status('Пакетная обработка...')
            system = self.ensure_system()
            model_path = normalize_windows_path(self.model_var.get().strip())
            if model_path and os.path.exists(model_path):
                system.load_model(model_path)
            max_images = int(self.max_images_var.get().strip())
            results = system.batch_process(
                test_color=self.test_color_var.get(),
                test_category=self.test_category_var.get(),
                max_images=max_images,
            )
            print(f'Пакетная обработка завершена. Обработано: {len(results)}')
        self.run_in_thread(task)


def main():
    root = tk.Tk()
    app = AdaptiveHalftoningUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()