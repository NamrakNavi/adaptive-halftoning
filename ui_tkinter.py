import os
import sys
import threading
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from config import Config
from system.AdaptiveHalftoningSystem import AdaptiveHalftoningSystem
from utils.file_utils import find_dataset_windows, normalize_windows_path


class AdaptiveHalftoningUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Adaptive Halftoning Studio')
        self.root.geometry('1280x860')
        self.root.minsize(1100, 760)

        self.config = Config()
        dataset_path = find_dataset_windows(self.config) or self.config.DATASET_PATH
        self.system = AdaptiveHalftoningSystem(dataset_path, self.config)

        self.image_path = tk.StringVar()
        self.model_path = tk.StringVar(value=self.config.MODEL_SAVE_PATH)
        self.dataset_path = tk.StringVar(value=dataset_path)
        self.status_var = tk.StringVar(value='Готово к работе')
        self.metrics_var = tk.StringVar(value='Метрики пока отсутствуют')
        self.tk_preview = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill='both', expand=True)

        top = ttk.LabelFrame(main, text='Параметры', padding=12)
        top.pack(fill='x', pady=(0, 10))

        ttk.Label(top, text='Изображение:').grid(row=0, column=0, sticky='w')
        ttk.Entry(top, textvariable=self.image_path, width=100).grid(row=0, column=1, sticky='ew', padx=8)
        ttk.Button(top, text='Открыть', command=self.select_image).grid(row=0, column=2)

        ttk.Label(top, text='Модель:').grid(row=1, column=0, sticky='w', pady=(8, 0))
        ttk.Entry(top, textvariable=self.model_path, width=100).grid(row=1, column=1, sticky='ew', padx=8, pady=(8, 0))
        ttk.Button(top, text='Выбрать', command=self.select_model).grid(row=1, column=2, pady=(8, 0))

        ttk.Label(top, text='Датасет:').grid(row=2, column=0, sticky='w', pady=(8, 0))
        ttk.Entry(top, textvariable=self.dataset_path, width=100).grid(row=2, column=1, sticky='ew', padx=8, pady=(8, 0))
        ttk.Button(top, text='Выбрать', command=self.select_dataset).grid(row=2, column=2, pady=(8, 0))

        btns = ttk.Frame(top)
        btns.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(12, 0))
        ttk.Button(btns, text='Загрузить модель', command=self.load_model).pack(side='left')
        ttk.Button(btns, text='Обработать изображение', command=self.process_image).pack(side='left', padx=8)

        top.columnconfigure(1, weight=1)

        center = ttk.Panedwindow(main, orient='vertical')
        center.pack(fill='both', expand=True)

        preview_box = ttk.Labelframe(center, text='Результат', padding=10)
        info_box = ttk.Labelframe(center, text='Информация', padding=10)
        center.add(preview_box, weight=5)
        center.add(info_box, weight=1)

        self.preview_label = ttk.Label(preview_box, text='После обработки здесь появится итоговая картинка', anchor='center')
        self.preview_label.pack(fill='both', expand=True)

        ttk.Label(info_box, textvariable=self.metrics_var, justify='left').pack(anchor='w')
        ttk.Label(info_box, textvariable=self.status_var, justify='left').pack(anchor='w', pady=(8, 0))

    def select_image(self):
        path = filedialog.askopenfilename(title='Выберите изображение', filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff')])
        if path:
            self.image_path.set(normalize_windows_path(path))

    def select_model(self):
        path = filedialog.askopenfilename(title='Выберите модель', filetypes=[('PyTorch model', '*.pth'), ('All files', '*.*')])
        if path:
            self.model_path.set(normalize_windows_path(path))

    def select_dataset(self):
        path = filedialog.askdirectory(title='Выберите папку датасета')
        if path:
            self.dataset_path.set(normalize_windows_path(path))
            self.config.update_dataset_path(self.dataset_path.get())
            self.system = AdaptiveHalftoningSystem(self.dataset_path.get(), self.config)

    def load_model(self):
        path = self.model_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror('Ошибка', 'Файл модели не найден')
            return
        ok = self.system.load_model(path)
        self.status_var.set('Модель загружена' if ok else 'Не удалось загрузить модель')

    def process_image(self):
        image_path = self.image_path.get().strip()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror('Ошибка', 'Выберите корректное изображение')
            return
        self.status_var.set('Обработка...')
        threading.Thread(target=self._worker, args=(image_path,), daemon=True).start()

    def _worker(self, image_path):
        try:
            model_path = self.model_path.get().strip()
            if model_path and os.path.exists(model_path):
                self.system.load_model(model_path)
            result = self.system.process_image(image_path, save_results=True)
            if result is None:
                self.root.after(0, lambda: self.status_var.set('Ошибка обработки'))
                return

            result_path = os.path.join(self.config.OUTPUT_DIR, f"{os.path.splitext(os.path.basename(image_path))[0]}_results.png")
            if os.path.exists(result_path):
                img = Image.open(result_path)
                img.thumbnail((1180, 620))
                self.tk_preview = ImageTk.PhotoImage(img)
                self.root.after(0, lambda: self.preview_label.configure(image=self.tk_preview, text=''))

            metrics_text = (
                f"SSIM ordered: {result['ssim_ordered']:.4f}
"
                f"SSIM error diffusion: {result['ssim_error']:.4f}
"
                f"SSIM adaptive: {result['ssim_adaptive']:.4f}
"
                f"SSIM stable preview: {result['ssim_adaptive_preview']:.4f}"
            )
            self.root.after(0, lambda: self.metrics_var.set(metrics_text))
            self.root.after(0, lambda: self.status_var.set('Готово'))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f'Ошибка: {e}'))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use('vista')
    except Exception:
        pass
    AdaptiveHalftoningUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
