import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Добавляем текущую папку в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from system.AdaptiveHalftoningSystem import AdaptiveHalftoningSystem
from utils.file_utils import (
    find_dataset_windows, 
    check_dataset_structure, 
    suggest_dataset_folders,
    create_dataset_structure,
    normalize_windows_path
)


def print_header():
    # Вывод заголовка программы
    print("\n" + "="*70)
    print(" " * 15 + "СИСТЕМА АДАПТИВНОГО РАСТРИРОВАНИЯ")
    print(" " * 20 + "(Версия для Windows)")
    print("="*70 + "\n")


def print_menu():
    # Вывод меню
    print("\nГЛАВНОЕ МЕНЮ:")
    print("1. Обучить новую модель")
    print("2. Обработать одно изображение")
    print("3. Пакетная обработка")
    print("4. Проверить структуру датасета")
    print("5. Создать структуру датасета")
    print("6. Настройки")
    print("0. Выход")
    
    return input("\nВыберите действие (0-6): ").strip()


def settings_menu(config):
    # Меню настроек
    while True:
        print("\n" + "="*50)
        print("НАСТРОЙКИ")
        print("="*50)
        print(f"1. Путь к датасету: {config.DATASET_PATH}")
        print(f"2. Папка для результатов: {config.OUTPUT_DIR}")
        print(f"3. Папка для моделей: {config.MODELS_DIR}")
        print(f"4. Batch Size: {config.BATCH_SIZE}")
        print(f"5. Learning Rate: {config.LEARNING_RATE}")
        print(f"6. Epochs: {config.EPOCHS}")
        print(f"7. Размер изображения: {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}")
        print("0. Назад")
        
        choice = input("\nВыберите параметр для изменения (0-7): ").strip()
        
        if choice == '1':
            new_path = input("Введите новый путь к датасету: ").strip()
            new_path = normalize_windows_path(new_path)
            if os.path.exists(new_path):
                config.update_dataset_path(new_path)
                print("✓ Путь обновлен")
            else:
                print("✗ Путь не существует!")
        
        elif choice == '2':
            new_path = input("Введите новую папку для результатов: ").strip()
            new_path = normalize_windows_path(new_path)
            config.update_output_dir(new_path)
        
        elif choice == '3':
            new_path = input("Введите новую папку для моделей: ").strip()
            new_path = normalize_windows_path(new_path)
            config.MODELS_DIR = new_path
            config.MODEL_SAVE_PATH = os.path.join(new_path, "adaptive_halftoning_model.pth")
            os.makedirs(new_path, exist_ok=True)
            print("✓ Папка для моделей обновлена")
        
        elif choice == '4':
            try:
                val = int(input(f"Batch Size (текущий {config.BATCH_SIZE}): "))
                if val > 0:
                    config.BATCH_SIZE = val
                    print("✓ Обновлено")
            except:
                print("✗ Неверное значение")
        
        elif choice == '5':
            try:
                val = float(input(f"Learning Rate (текущий {config.LEARNING_RATE}): "))
                if val > 0:
                    config.LEARNING_RATE = val
                    print("✓ Обновлено")
            except:
                print("✗ Неверное значение")
        
        elif choice == '6':
            try:
                val = int(input(f"Epochs (текущий {config.EPOCHS}): "))
                if val > 0:
                    config.EPOCHS = val
                    print("✓ Обновлено")
            except:
                print("✗ Неверное значение")
        
        elif choice == '7':
            try:
                w = int(input("Ширина: "))
                h = int(input("Высота: "))
                if w > 0 and h > 0:
                    config.IMG_SIZE = (w, h)
                    print("✓ Обновлено")
            except:
                print("✗ Неверное значение")
        
        elif choice == '0':
            break


def main():
    # Основная функция
    print_header()
    
    # Инициализация конфигурации
    config = Config()
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Папка датасета: {config.DATASET_PATH}")
    print(f"Папка результатов: {config.OUTPUT_DIR}")
    
    system = None
    
    while True:
        choice = print_menu()
        
        if choice == '1':
            # Обучение модели
            print("\n" + "="*50)
            print("ОБУЧЕНИЕ МОДЕЛИ")
            print("="*50)
            
            # Поиск датасета
            dataset_path = find_dataset_windows(config)
            
            if dataset_path is None:
                print("✗ Датасет не найден. Обучение невозможно.")
                continue
            
            config.update_dataset_path(dataset_path)
            
            # Проверка структуры
            structure_ok, stats = check_dataset_structure(dataset_path)
            
            # Выбор параметров
            print("\nПараметры обучения:")
            train_color = input("Цвет для обучения (all/colored/black and white) [all]: ").strip()
            if train_color not in ['all', 'colored', 'black and white']:
                train_color = 'all'
            
            train_category = input("Категория для обучения (all/nature/transport) [all]: ").strip()
            if train_category not in ['all', 'nature', 'transport']:
                train_category = 'all'
            
            # Инициализация системы
            print("\nИнициализация системы...")
            system = AdaptiveHalftoningSystem(dataset_path, config)
            
            # Создание DataLoader
            train_loader, _ = system.create_dataloaders(
                train_color=train_color,
                train_category=train_category
            )
            
            if train_loader and len(train_loader) > 0:
                # Обучение
                epochs = input(f"Количество эпох (Enter для {config.EPOCHS}): ").strip()
                epochs = int(epochs) if epochs.isdigit() else config.EPOCHS
                
                system.train(train_loader, epochs=epochs)
                
                # Сохранение модели
                save_path = input(f"Сохранить модель как [{config.MODEL_SAVE_PATH}]: ").strip()
                if save_path:
                    system.save_model(save_path)
                else:
                    system.save_model()
            else:
                print("✗ Не удалось создать DataLoader. Проверьте датасет.")
        
        elif choice == '2':
            # Обработка одного изображения
            print("\n" + "="*50)
            print("ОБРАБОТКА ИЗОБРАЖЕНИЯ")
            print("="*50)
            
            # Запрос пути к изображению
            image_path = input("Введите путь к изображению: ").strip()
            image_path = normalize_windows_path(image_path)
            
            if not os.path.exists(image_path):
                print(f"✗ Файл не найден: {image_path}")
                continue
            
            # Инициализация системы, если нужно
            if system is None:
                dataset_path = find_dataset_windows(config)
                if dataset_path:
                    system = AdaptiveHalftoningSystem(dataset_path, config)
                else:
                    # Временная инициализация без датасета
                    system = AdaptiveHalftoningSystem(config.DATASET_PATH, config)
            
            # Загрузка модели
            model_path = input("Путь к модели (Enter если нет): ").strip()
            if model_path and os.path.exists(model_path):
                system.load_model(model_path)
            elif os.path.exists(config.MODEL_SAVE_PATH):
                system.load_model(config.MODEL_SAVE_PATH)
            
            # Обработка
            result = system.process_image(image_path, save_results=True)
            
            if result:
                print("\n" + "="*50)
                print("РЕЗУЛЬТАТЫ:")
                print(f"Упорядоченное: SSIM = {result['ssim_ordered']:.4f}")
                print(f"Error Diffusion: SSIM = {result['ssim_error']:.4f}")
                print(f"Адаптивное: SSIM = {result['ssim_adaptive']:.4f}")
                print("="*50)
        
        elif choice == '3':
            # Пакетная обработка
            print("\n" + "="*50)
            print("ПАКЕТНАЯ ОБРАБОТКА")
            print("="*50)
            
            # Поиск датасета
            dataset_path = find_dataset_windows(config)
            
            if dataset_path is None:
                print("✗ Датасет не найден.")
                continue
            
            # Инициализация системы
            if system is None:
                system = AdaptiveHalftoningSystem(dataset_path, config)
            
            # Загрузка модели
            if os.path.exists(config.MODEL_SAVE_PATH):
                system.load_model(config.MODEL_SAVE_PATH)
            
            # Выбор параметров
            test_color = input("Цвет для тестирования (all/colored/black and white) [all]: ").strip()
            if test_color not in ['all', 'colored', 'black and white']:
                test_color = 'all'
            
            test_category = input("Категория для тестирования (all/nature/transport) [all]: ").strip()
            if test_category not in ['all', 'nature', 'transport']:
                test_category = 'all'
            
            max_images = input("Максимум изображений для обработки [20]: ").strip()
            max_images = int(max_images) if max_images.isdigit() else 20
            
            # Обработка
            results = system.batch_process(
                test_color=test_color,
                test_category=test_category,
                max_images=max_images
            )
            
            print(f"\nОбработано {len(results)} изображений")
        
        elif choice == '4':
            # Проверка структуры датасета
            print("\n" + "="*50)
            print("ПРОВЕРКА СТРУКТУРЫ ДАТАСЕТА")
            print("="*50)
            
            dataset_path = input("Введите путь к датасету (Enter для текущего): ").strip()
            if not dataset_path:
                dataset_path = config.DATASET_PATH
            
            dataset_path = normalize_windows_path(dataset_path)
            
            if os.path.exists(dataset_path):
                structure_ok, stats = check_dataset_structure(dataset_path)
                
                # Подсчет общего количества изображений
                total_images = 0
                for mode in stats:
                    for color in stats[mode]:
                        for category in stats[mode][color]:
                            total_images += stats[mode][color][category]
                
                print(f"\nВсего изображений: {total_images}")
                
                if total_images == 0:
                    print("\n⚠ В датасете нет изображений!")
                    create = input("Создать структуру датасета? (y/n): ").lower()
                    if create == 'y':
                        create_dataset_structure(dataset_path)
            else:
                print(f"✗ Путь не существует: {dataset_path}")
                create = input("Создать структуру датасета? (y/n): ").lower()
                if create == 'y':
                    create_dataset_structure(dataset_path)
        
        elif choice == '5':
            # Создание структуры датасета
            print("\n" + "="*50)
            print("СОЗДАНИЕ СТРУКТУРЫ ДАТАСЕТА")
            print("="*50)
            
            base_path = input("Введите путь для создания датасета (Enter для текущей папки): ").strip()
            if not base_path:
                base_path = os.getcwd()
            
            base_path = normalize_windows_path(base_path)
            
            if create_dataset_structure(base_path):
                print("\n✓ Структура создана!")
                print(f"Путь: {base_path}")
                print("\nТеперь поместите изображения в соответствующие папки:")
                print("  train/colored/nature/     - цветные изображения природы для обучения")
                print("  train/colored/transport/  - цветные изображения транспорта для обучения")
                print("  train/black and white/nature/ - ч/б изображения природы для обучения")
                print("  и так далее...")
        
        elif choice == '6':
            # Настройки
            settings_menu(config)
        
        elif choice == '0':
            print("\nПрограмма завершена.")
            break
        
        else:
            print("\n✗ Неверный выбор. Попробуйте снова.")
        
        input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*50)
        print("УСТРАНЕНИЕ НЕПОЛАДОК:")
        print("1. Убедитесь, что все зависимости установлены:")
        print("   pip install torch torchvision numpy matplotlib pillow scikit-image")
        print("2. Проверьте пути к файлам")
        print("3. Убедитесь, что изображения существуют")
        print("="*50)
        
        input("\nНажмите Enter для выхода...")