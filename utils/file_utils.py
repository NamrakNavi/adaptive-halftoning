import os
import sys
from pathlib import Path

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f" Создана директория: {directory}")
    return directory


def get_windows_drives():
    # Получение списка доступных дисков в Windows
    drives = []
    try:
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append(drive)
    except:
        drives = ['C:\\', 'D:\\', 'E:\\', 'F:\\']
    
    return drives


def normalize_windows_path(path):
    # Нормализация пути для Windows
    if not path:
        return path
    
    path = path.replace('/', '\\')
    path = path.strip()
    
    if len(path) == 2 and path[1] == ':':
        path += '\\'
    
    return path


def find_dataset_windows(config):
    """Поиск датасета в Windows"""
    print("=" * 60)
    print("ПОИСК ДАТАСЕТА В WINDOWS")
    print("=" * 60)
    
    current_dir = os.getcwd()
    
    possible_paths = [
        config.DATASET_PATH,
        os.path.join(current_dir, "halftoning_dataset"),
        os.path.join(current_dir, "dataset", "halftoning_dataset"),
        os.path.join(current_dir, "data", "halftoning_dataset"),
        os.path.join(os.path.expanduser("~"), "halftoning_dataset"),
        os.path.join(os.path.expanduser("~"), "Documents", "halftoning_dataset"),
        "C:\\halftoning_dataset",
        "D:\\halftoning_dataset",
        "E:\\halftoning_dataset",
    ]
    
    print("Поиск в стандартных расположениях...")
    
    for path in possible_paths:
        if path and os.path.exists(path):
            if os.path.exists(os.path.join(path, 'train')) or os.path.exists(os.path.join(path, 'test')):
                print(f"✓ НАЙДЕН: {path}")
                return path
            else:
                images = find_images_in_folder(path, max_count=5)
                if images:
                    print(f"✓ НАЙДЕН: {path}")
                    return path
    
    print("\nДатасет не найден в стандартных путях")
    
    custom_path = input("Введите полный путь к датасету (или 'q' для выхода): ").strip()
    
    if custom_path.lower() == 'q':
        return None
    
    custom_path = normalize_windows_path(custom_path)
    
    if os.path.exists(custom_path):
        print(f"✓ Путь существует: {custom_path}")
        return custom_path
    else:
        print("✗ Путь не существует!")
        return None


def find_images_in_folder(folder_path, extensions=None, max_count=None):
    """Поиск изображений в папке и подпапках"""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
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
    """Проверка структуры датасета"""
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
            print(f"  ⚠ Отсутствует папка: {mode}/")
            structure_ok = False
            continue
        
        stats[mode] = {}
        
        for color_type in required_structure[mode]:
            color_path = os.path.join(mode_path, color_type)
            if not os.path.exists(color_path):
                print(f"  ⚠ Отсутствует папка: {mode}/{color_type}/")
                structure_ok = False
                continue
            
            stats[mode][color_type] = {}
            
            for category in categories:
                category_path = os.path.join(color_path, category)
                if os.path.exists(category_path):
                    images = find_images_in_folder(category_path)
                    count = len(images)
                    stats[mode][color_type][category] = count
                    if count > 0:
                        print(f"  ✓ {mode}/{color_type}/{category}/: {count} изображений")
                    else:
                        print(f"  ⚠ {mode}/{color_type}/{category}/: пусто")
                else:
                    stats[mode][color_type][category] = 0
                    print(f"  ⚠ Отсутствует папка: {mode}/{color_type}/{category}/")
    
    return structure_ok, stats


def create_dataset_structure(base_path):
    """Создание структуры датасета"""
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
        return True
    
    return False


def get_image_files(directory, extensions=None):
    """Получение списка файлов изображений в директории"""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files