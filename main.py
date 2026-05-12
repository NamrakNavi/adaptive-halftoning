import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from system.AdaptiveHalftoningSystem import AdaptiveHalftoningSystem
from utils.file_utils import find_dataset_windows, check_dataset_structure, create_dataset_structure, normalize_windows_path


def print_header():
    print(" " + "=" * 70)
    print(" " * 15 + "СИСТЕМА АДАПТИВНОГО РАСТРИРОВАНИЯ")
    print(" " * 16 + "(Стабилизированная версия)")
    print("=" * 70 + " ")


def print_menu():
    print("ГЛАВНОЕ МЕНЮ:")
    print("1. Обучить новую модель")
    print("2. Обработать одно изображение")
    print("0. Выход")
    return input("Выберите действие (0-2): ").strip()


def main():
    print_header()
    config = Config()
    system = None

    while True:
        choice = print_menu()
        if choice == '1':
            dataset_path = find_dataset_windows(config)
            if dataset_path is None:
                print('Датасет не найден. Обучение невозможно.')
                continue
            config.update_dataset_path(dataset_path)
            system = AdaptiveHalftoningSystem(dataset_path, config)
            train_loader, _ = system.create_dataloaders()
            if train_loader and len(train_loader) > 0:
                system.train(train_loader, epochs=config.EPOCHS)
                system.save_model()
        elif choice == '2':
            image_path = normalize_windows_path(input('Введите путь к изображению: ').strip())
            if not os.path.exists(image_path):
                print(f'Файл не найден: {image_path}')
                continue
            if system is None:
                dataset_path = find_dataset_windows(config) or config.DATASET_PATH
                system = AdaptiveHalftoningSystem(dataset_path, config)
            if os.path.exists(config.MODEL_SAVE_PATH):
                system.load_model(config.MODEL_SAVE_PATH)
            result = system.process_image(image_path, save_results=True)
            if result:
                print(" " + "=" * 50)
                print("РЕЗУЛЬТАТЫ:")
                print(f"Упорядоченное: SSIM = {result['ssim_ordered']:.4f}")
                print(f"Error Diffusion: SSIM = {result['ssim_error']:.4f}")
                print(f"Адаптивное: SSIM = {result['ssim_adaptive']:.4f}")
                print(f"Стабильная картинка: SSIM = {result['ssim_adaptive_preview']:.4f}")
                print("=" * 50)
        elif choice == '0':
            print('Программа завершена.')
            break
        else:
            print('✗ Неверный выбор. Попробуйте снова.')


if __name__ == '__main__':
    main()