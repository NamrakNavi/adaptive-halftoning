import os


class Config:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_PATH = os.path.join(self.BASE_DIR, 'halftoning_dataset')
        self.MODELS_DIR = os.path.join(self.BASE_DIR, 'models', 'saved')
        self.MODEL_SAVE_PATH = os.path.join(self.MODELS_DIR, 'adaptive_halftoning_model.pth')
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, 'results')
        self.LOGS_DIR = os.path.join(self.OUTPUT_DIR, 'logs')
        self.PLOTS_DIR = os.path.join(self.OUTPUT_DIR, 'plots')
        self.METRICS_DIR = os.path.join(self.OUTPUT_DIR, 'metrics')

        self.PATCH_SIZE = 8
        self.BATCH_SIZE = 16
        self.MIN_BATCH_SIZE = 2
        self.EPOCHS = 60
        self.LEARNING_RATE = 0.0003
        self.IMG_SIZE = (256, 256)
        self.IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']

        self.ADAPTIVE_STRENGTH = 0.55
        self.GAUSSIAN_SIGMA = 1.0
        self.LOCAL_WINDOW = 7
        self.MIN_THRESHOLD = 0.20
        self.MAX_THRESHOLD = 0.80
        self.MIX_WITH_ERROR_DIFFUSION = 0.35
        self.ENABLE_MEDIAN_POST = True

        self.ENABLE_OPTIONAL_METRICS = True
        self.ACTIVE_METRICS = ['SSIM', 'PSNR', 'MSE', 'IoU', 'LPIPS', 'BRISQUE', 'NIQE']
        self.BATCH_METRICS = ['FID']

        self._create_directories()

    def _create_directories(self):
        directories = [
            self.MODELS_DIR,
            self.OUTPUT_DIR,
            self.LOGS_DIR,
            self.PLOTS_DIR,
            self.METRICS_DIR,
            os.path.join(self.DATASET_PATH, 'train', 'colored', 'nature'),
            os.path.join(self.DATASET_PATH, 'train', 'colored', 'transport'),
            os.path.join(self.DATASET_PATH, 'train', 'black and white', 'nature'),
            os.path.join(self.DATASET_PATH, 'train', 'black and white', 'transport'),
            os.path.join(self.DATASET_PATH, 'test', 'colored', 'nature'),
            os.path.join(self.DATASET_PATH, 'test', 'colored', 'transport'),
            os.path.join(self.DATASET_PATH, 'test', 'black and white', 'nature'),
            os.path.join(self.DATASET_PATH, 'test', 'black and white', 'transport'),
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def update_dataset_path(self, new_path):
        self.DATASET_PATH = new_path
        print(f'Путь к датасету обновлен: {new_path}')

    def update_output_dir(self, new_dir):
        self.OUTPUT_DIR = new_dir
        self.LOGS_DIR = os.path.join(self.OUTPUT_DIR, 'logs')
        self.PLOTS_DIR = os.path.join(self.OUTPUT_DIR, 'plots')
        self.METRICS_DIR = os.path.join(self.OUTPUT_DIR, 'metrics')
        for directory in [self.OUTPUT_DIR, self.LOGS_DIR, self.PLOTS_DIR, self.METRICS_DIR]:
            os.makedirs(directory, exist_ok=True)
        print(f'Директория для результатов обновлена: {new_dir}')
