import os
import sys
import threading
import traceback
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from system.AdaptiveHalftoningSystem import AdaptiveHalftoningSystem
from utils.file_utils import find_dataset_windows, normalize_windows_path, create_dataset_structure, check_dataset_structure


class TeeRedirector:
    def __init__(self, widget, logfile_getter):
        self.widget = widget
        self.logfile_getter = logfile_getter

    def write(self, text):
        if not text:
            return
        try:
            self.widget.after(0, self._append_to_widget, text)
        except Exception:
            pass
        try:
            log_path = self.logfile_getter()
            if log_path:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(text)
        except Exception:
            pass

    def flush(self):
        pass

    def _append_to_widget(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)


class AdaptiveHalftoningUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Система адаптивного растрирования')
        self.root.geometry('1120x780')
        self.config = Config()
        self.system = None
        self.current_log_file = None
        self._build_ui()
        self._init_defaults()
        self._redirect_output()
        self.write_log('UI инициализирован.')

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(main, text='Пути и параметры', padding=10)
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
        self.log_file_var = tk.StringVar()
        self.train_color_var = tk.StringVar(value='all')
        self.train_category_var = tk.StringVar(value='all')
        self.test_color_var = tk.StringVar(value='all')
        self.test_category_var = tk.StringVar(value='all')

        self._path_row(top, 0, 'Датасет:', self.dataset_var, self.choose_dataset)
        self._path_row(top, 1, 'Результаты:', self.output_var, self.choose_output_dir)
        self._path_row(top, 2, 'Модель:', self.model_var, self.choose_model_file)
        self._path_row(top, 3, 'Изображение:', self.image_var, self.choose_image)
        self._path_row(top, 4, 'Лог-файл:', self.log_file_var, self.choose_log_file)

        params = ttk.Frame(top)
        params.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        for i in range(8):
            params.columnconfigure(i, weight=1)

        ttk.Label(params, text='Epochs').grid(row=0, column=0, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.epochs_var).grid(row=0, column=1, sticky='ew', padx=4, pady=4)
        ttk.Label(params, text='Batch Size').grid(row=0, column=2, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.batch_var).grid(row=0, column=3, sticky='ew', padx=4, pady=4)
        ttk.Label(params, text='Learning Rate').grid(row=0, column=4, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.lr_var).grid(row=0, column=5, sticky='ew', padx=4, pady=4)
        ttk.Label(params, text='Макс. изображений').grid(row=0, column=6, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.max_images_var).grid(row=0, column=7, sticky='ew', padx=4, pady=4)

        ttk.Label(params, text='Ширина').grid(row=1, column=0, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.img_w_var).grid(row=1, column=1, sticky='ew', padx=4, pady=4)
        ttk.Label(params, text='Высота').grid(row=1, column=2, sticky='w', padx=4, pady=4)
        ttk.Entry(params, textvariable=self.img_h_var).grid(row=1, column=3, sticky='ew', padx=4, pady=4)

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
        buttons = [
            ('Найти датасет', self.auto_find_dataset),
            ('Проверить датасет', self.check_dataset),
            ('Создать структуру датасета', self.create_dataset),
            ('Инициализировать систему', self.init_system),
            ('Обучить модель', self.train_model),
            ('Загрузить модель', self.load_model),
            ('Обработать изображение', self.process_image),
            ('Пакетная обработка', self.batch_process),
            ('Очистить лог', self.clear_log),
        ]
        for idx, (text, cmd) in enumerate(buttons):
            ttk.Button(actions, text=text, command=cmd).grid(row=idx // 3, column=idx % 3, sticky='ew', padx=6, pady=6)
        for i in range(3):
            actions.columnconfigure(i, weight=1)

        log_frame = ttk.LabelFrame(main, text='Логи окна', padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_widget = tk.Text(log_frame, wrap='word', height=24)
        self.log_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_widget.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_widget.configure(yscrollcommand=scroll.set)

        self.status_var = tk.StringVar(value='Готово')
        ttk.Label(main, textvariable=self.status_var).pack(fill=tk.X)

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
        default_log = os.path.join(self.config.OUTPUT_DIR, 'ui_session.log')
        self.log_file_var.set(default_log)
        self.current_log_file = default_log

    def _redirect_output(self):
        redirector = TeeRedirector(self.log_widget, self.get_log_file_path)
        sys.stdout = redirector
        sys.stderr = redirector

    def get_log_file_path(self):
        path = self.log_file_var.get().strip()
        if not path:
            return self.current_log_file
        self.current_log_file = path
        return self.current_log_file

    def write_log(self, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{timestamp}] {message}')

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()
        self.write_log(f'Статус: {text}')

    def clear_log(self):
        self.log_widget.delete('1.0', tk.END)
        self.write_log('Лог очищен.')

    def choose_dataset(self):
        path = filedialog.askdirectory(title='Выберите папку датасета')
        if path:
            self.dataset_var.set(normalize_windows_path(path))
            self.write_log(f'Выбран датасет: {path}')

    def choose_output_dir(self):
        path = filedialog.askdirectory(title='Выберите папку результатов')
        if path:
            self.output_var.set(normalize_windows_path(path))
            self.write_log(f'Выбрана папка результатов: {path}')

    def choose_model_file(self):
        path = filedialog.askopenfilename(title='Выберите файл модели', filetypes=[('PyTorch model', '*.pth'), ('All files', '*.*')])
        if path:
            self.model_var.set(normalize_windows_path(path))
            self.write_log(f'Выбран файл модели: {path}')

    def choose_image(self):
        path = filedialog.askopenfilename(title='Выберите изображение', filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff'), ('All files', '*.*')])
        if path:
            self.image_var.set(normalize_windows_path(path))
            self.write_log(f'Выбрано изображение: {path}')

    def choose_log_file(self):
        path = filedialog.asksaveasfilename(title='Выберите файл логов', defaultextension='.txt', filetypes=[('Text log', '*.txt'), ('Log files', '*.log'), ('All files', '*.*')])
        if path:
            self.log_file_var.set(normalize_windows_path(path))
            self.current_log_file = normalize_windows_path(path)
            self.write_log(f'Логи будут сохраняться в: {path}')

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
            self.write_log('AdaptiveHalftoningSystem создан.')
        return self.system

    def run_in_thread(self, target, title):
        self.write_log(f'Запуск задачи: {title}')
        threading.Thread(target=self._safe_run, args=(target, title), daemon=True).start()

    def _safe_run(self, target, title):
        try:
            target()
            self.set_status('Готово')
            self.write_log(f'Задача завершена: {title}')
        except Exception as e:
            self.set_status('Ошибка')
            self.write_log(f'Ошибка в задаче {title}: {e}')
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror('Ошибка', str(e)))

    def auto_find_dataset(self):
        def task():
            self.set_status('Поиск датасета...')
            path = find_dataset_windows(self.config)
            if path:
                self.dataset_var.set(normalize_windows_path(path))
                self.write_log(f'Найден датасет: {path}')
            else:
                self.write_log('Датасет не найден автоматически.')
        self.run_in_thread(task, 'Поиск датасета')

    def check_dataset(self):
        def task():
            self.set_status('Проверка датасета...')
            self.sync_config_from_ui()
            structure_ok, stats = check_dataset_structure(self.config.DATASET_PATH)
            self.write_log(f'Структура датасета корректна: {structure_ok}')
            self.write_log(f'Статистика датасета: {stats}')
        self.run_in_thread(task, 'Проверка датасета')

    def create_dataset(self):
        def task():
            self.set_status('Создание датасета...')
            self.sync_config_from_ui()
            ok = create_dataset_structure(self.config.DATASET_PATH)
            self.write_log('Структура датасета создана.' if ok else 'Не удалось создать структуру датасета.')
        self.run_in_thread(task, 'Создание датасета')

    def init_system(self):
        def task():
            self.set_status('Инициализация системы...')
            self.system = None
            self.ensure_system()
        self.run_in_thread(task, 'Инициализация системы')

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
                self.write_log('Train DataLoader пуст. Обучение остановлено.')
                return
            self.write_log(f'Train DataLoader создан. Батчей: {len(train_loader)}')
            if test_loader is not None:
                self.write_log(f'Test DataLoader создан. Батчей: {len(test_loader)}')
            system.train(train_loader, epochs=self.config.EPOCHS)
            system.save_model(self.config.MODEL_SAVE_PATH)
            self.model_var.set(self.config.MODEL_SAVE_PATH)
            self.write_log(f'Модель сохранена: {self.config.MODEL_SAVE_PATH}')
        self.run_in_thread(task, 'Обучение модели')

    def load_model(self):
        def task():
            self.set_status('Загрузка модели...')
            system = self.ensure_system()
            model_path = normalize_windows_path(self.model_var.get().strip())
            if not model_path or not os.path.exists(model_path):
                self.write_log(f'Файл модели не найден: {model_path}')
                return
            system.load_model(model_path)
            self.write_log('Модель успешно загружена.')
        self.run_in_thread(task, 'Загрузка модели')

    def process_image(self):
        def task():
            self.set_status('Выбор изображения для обработки...')
            chosen = filedialog.askopenfilename(
                title='Выберите изображение для обработки',
                filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff'), ('All files', '*.*')]
            )
            if not chosen:
                self.write_log('Пользователь отменил выбор изображения.')
                return
            image_path = normalize_windows_path(chosen)
            self.image_var.set(image_path)
            self.write_log(f'Выбрано изображение: {image_path}')
            system = self.ensure_system()
            model_path = normalize_windows_path(self.model_var.get().strip())
            if model_path and os.path.exists(model_path):
                system.load_model(model_path)
            result = system.process_image(image_path, save_results=True)
            if result:
                self.write_log(
                    f"Результаты: ordered={result.get('ssim_ordered', 0):.4f}, "
                    f"error={result.get('ssim_error', 0):.4f}, dbs={result.get('ssim_dbs', 0):.4f}, "
                    f"diff={result.get('ssim_diff', 0):.4f}, adaptive={result.get('ssim_adaptive', 0):.4f}, "
                    f"preview={result.get('ssim_preview', 0):.4f}"
                )
        self.run_in_thread(task, 'Обработка изображения')

    def batch_process(self):
        def task():
            self.set_status('Пакетная обработка...')
            system = self.ensure_system()
            model_path = normalize_windows_path(self.model_var.get().strip())
            if model_path and os.path.exists(model_path):
                system.load_model(model_path)
            results = system.batch_process(
                test_color=self.test_color_var.get(),
                test_category=self.test_category_var.get(),
                max_images=int(self.max_images_var.get().strip())
            )
            self.write_log(f'Пакетная обработка завершена. Обработано: {len(results)}')
        self.run_in_thread(task, 'Пакетная обработка')


def main():
    root = tk.Tk()
    AdaptiveHalftoningUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()