import os
import sys
import warnings
import traceback

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_tkinter_ui():
    try:
        from ui_tkinter import main as ui_main
        ui_main()
    except ImportError as e:
        print(f'Ошибка импорта UI: {e}')
        print('Убедитесь, что файл ui_tkinter.py находится в корне проекта.')
        raise


def main():
    run_tkinter_ui()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Программа прервана пользователем')
    except Exception as e:
        print(f'Произошла ошибка: {e}')
        traceback.print_exc()
        print('' + '=' * 60)
        print('УСТРАНЕНИЕ НЕПОЛАДОК:')
        print('1. Проверьте наличие файлов ui_tkinter.py, config.py и папок system/, algorithms/, data/, models/, utils/.')
        print('2. Убедитесь, что установлены зависимости:')
        print('   pip install torch torchvision numpy matplotlib pillow scikit-image scipy')
        print('3. Проверьте корректность файла models/unet.py и наличие __init__.py в пакетах.')
        print('=' * 60)
        input('Нажмите Enter для выхода...')