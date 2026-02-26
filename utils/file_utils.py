import os
import sys
import subprocess
from pathlib import Path

def get_windows_drives():
    # Получение списка доступных дисков в Windows
    drives = []
    try:
        # Для Windows используем буквы дисков от A до Z
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append(drive)
    except:
        # Если не удалось получить список дисков, возвращаем стандартные
        drives = ['C:\\', 'D:\\', 'E:\\', 'F:\\']

    return drives

def normalize_windows_path(path):
    # Нормализация пути для Windows
    if not path:
        return path

    # Заменяем прямые слеши на обратные
    path = path.replace('/', '\\')

    # Убираем лишние пробелы
    path = path.strip()

    # Добавляем обратный слеш в конце, если это диск
    if len(path) == 2 and path[1] == ':':
        path += '\\'

    return path

def find_dataset_windows(config):
    # Поиск датасета в Windows

    print("=" * 60)
    print("ПОИСК ДАТАСЕТА В WINDOWS")
    print("=" * 60)

    # Текущая директория
    current_dir = os.getcwd()

    # Возможные пути для поиска датасета
    possible_paths = [
        # Относительные пути
        config.DATASET_PATH,
        os.path.join(current_dir, "halftoning_dataset"),
        os.path.join(current_dir, "dataset", "halftoning_dataset"),
        os.path.join(current_dir, "data", "halftoning_dataset"),

        # Пути в домашней директории
        os.path.join(os.path.expanduser("~"), "halftoning_dataset"),
        os.path.join(os.path.expanduser("~"), "Documents", "halftoning_dataset"),
        os.path.join(os.path.expanduser("~"), "Downloads", "halftoning_dataset"),

        # Пути на разных дисках
        "C:\\halftoning_dataset",
        "C:\\datasets\\halftoning_dataset",
        "C:\\Users\\Public\\halftoning_dataset",
        "D:\\halftoning_dataset",
        "D:\\datasets\\halftoning_dataset",
        "E:\\halftoning_dataset",
        "F:\\halftoning_dataset",
    ]

    print("Поиск в стандартных расположениях...")

    for path in possible_paths:
        if path and os.path.exists(path):
            # Проверяем, что в папке есть подпапки train/test
            if os.path.exists(os.path.join(path, 'train')) or os.path.exists(os.path.join(path, 'test')):
                print(f"НАЙДЕН: {path}")
                return path
            else:
                # Если нет train/test, но есть изображения - тоже подходит
                images = find_images_in_folder(path, max_count=5)
                if images:
                    print(f"НАЙДЕН (содержит изображения): {path}")
                    return path

    # Если не нашли, показываем пользователю структуру дисков
    print("\n" + "=" * 60)
    print("ДАТАСЕТ НЕ НАЙДЕН В СТАНДАРТНЫХ ПУТЯХ")
    print("=" * 60)

    print("\nДоступные диски:")
    drives = get_windows_drives()
    for drive in drives:
        print(f"  📁 {drive}")

        # Показываем первые несколько папок на каждом диске
        try:
            items = []
            for item in os.listdir(drive):
                if os.path.isdir(os.path.join(drive, item)):
                    items.append(item)
                    if len(items) >= 5:
                        break

            if items:
                print(f"     Папки: {', '.join(items[:5])}")
                if len(items) > 5:
                    print(f"     ... и еще {len(items)-5}")
        except:
            pass

    print("\nТекущая директория:")
    print(f"  📁 {current_dir}")

    # Показываем папки в текущей директории
    try:
        folders = []
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                folders.append(item)

        if folders:
            print("\nПапки в текущей директории:")
            for folder in folders[:10]:
                print(f"  📁 {folder}/")
            if len(folders) > 10:
                print(f"  ... и еще {len(folders)-10} папок")
    except:
        pass

    # Запрашиваем путь у пользователя
    print("\n" + "=" * 60)
    custom_path = input("Введите полный путь к папке с датасетом (или 'q' для выхода): ").strip()

    if custom_path.lower() == 'q':
        return None

    # Нормализуем путь
    custom_path = normalize_windows_path(custom_path)

    if os.path.exists(custom_path):
        print(f" Путь существует: {custom_path}")

        # Проверяем, есть ли там изображения
        images = find_images_in_folder(custom_path, max_count=10)
        if images:
            print(f" Найдено изображений: {len(images)}")
            print(f"  Пример: {os.path.basename(images[0])}")
            return custom_path
        else:
            print(" В папке нет изображений!")
            retry = input("Продолжить всё равно? (y/n): ").lower()
            if retry == 'y':
                return custom_path
            else:
                return None
    else:
        print("Путь не существует!")
        return None

def find_images_in_folder(folder_path, extensions=None, max_count=None):
    # Поиск изображений в папке и подпапках
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif']

    images = []
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    images.append(os.path.join(root, file))
                    if max_count and len(images) >= max_count:
                        return images
    except Exception as e:
        print(f"Ошибка при поиске: {e}")

    return images

def check_dataset_structure(dataset_path):
    # Проверка структуры датасета
    print(f"\nПроверка структуры датасета: {dataset_path}")

    required_structure = {
        'train': ['colored', 'black and white'],
        'test': ['colored', 'black and white']
    }

    categories = ['nature', 'transport']

    structure_ok = True
    stats = {}

    for mode in required_structure:
        mode_path = os.path.join(dataset_path, mode)
        if not os.path.exists(mode_path):
            print(f"Отсутствует папка: {mode}/")
            structure_ok = False
            continue

        stats[mode] = {}

        for color_type in required_structure[mode]:
            color_path = os.path.join(mode_path, color_type)
            if not os.path.exists(color_path):
                print(f"Отсутствует папка: {mode}/{color_type}/")
                structure_ok = False
                continue

            stats[mode][color_type] = {}

            # Считаем изображения в категориях
            for category in categories:
                category_path = os.path.join(color_path, category)
                if os.path.exists(category_path):
                    images = find_images_in_folder(category_path)
                    count = len(images)
                    stats[mode][color_type][category] = count
                    if count > 0:
                        print(f"{mode}/{color_type}/{category}/: {count} изображений")
                    else:
                        print(f"{mode}/{color_type}/{category}/: пусто")
                else:
                    stats[mode][color_type][category] = 0
                    print(f"Отсутствует папка: {mode}/{color_type}/{category}/")

    return structure_ok, stats

def suggest_dataset_folders():
    # Предложение по созданию структуры датасета
    print("\n" + "=" * 60)
    print("РЕКОМЕНДУЕМАЯ СТРУКТУРА ДАТАСЕТА:")
    print("=" * 60)
    print("""
halftoning_dataset/
├── train/
│   ├── colored/
│   │   ├── nature/          # изображения природы
│   │   └── transport/       # изображения транспорта
│   └── black and white/
│       ├── nature/          # ч/б изображения природы
│       └── transport/       # ч/б изображения транспорта
└── test/
    ├── colored/
    │   ├── nature/
    │   └── transport/
    └── black and white/
        ├── nature/
        └── transport/
    """)

    print("\nВы можете:")
    print("1. Создать эту структуру в текущей папке")
    print("2. Указать путь к существующему датасету")
    print("3. Использовать программу без строгой структуры (будет искать все изображения)")

    return input("\nВыберите действие (1/2/3): ").strip()

def create_dataset_structure(base_path):
    # Создание структуры датасета
    base_path = normalize_windows_path(base_path)

    modes = ['train', 'test']
    color_types = ['colored', 'black and white']
    categories = ['nature', 'transport']

    created = []

    for mode in modes:
        for color in color_types:
            for category in categories:
                folder = os.path.join(base_path, mode, color, category)
                try:
                    os.makedirs(folder, exist_ok=True)
                    created.append(folder)
                except Exception as e:
                    print(f"Ошибка при создании {folder}: {e}")

    if created:
        print(f"\n✓ Создано {len(created)} папок")
        print(f"Базовая папка: {base_path}")
        print("\nПоместите изображения в соответствующие папки:")
        for folder in created[:3]:  # Показываем первые 3
            print(f"  📁 {folder}")

        return True
    return False