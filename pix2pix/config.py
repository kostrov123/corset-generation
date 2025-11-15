import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

TRAIN_GENERATOR_ONLY = True
LOAD_GENERATOR_MODEL = False
SAVE_MODEL = True

IM_SIZE = 256
LR = 2e-4
LR_PATIENCE = 25 # Через сколько эпох уменьшать LR, если не улучшаются метрики
BATCH_SIZE = 45
NUM_EPOCHS = 2000 # Число эпох
GAN_LAMBDA = 0 if TRAIN_GENERATOR_ONLY else 1 # Для первой стадии множитель у ГАН равен 0, для второй 1
L1_LAMBDA = 300
L2_LAMBDA = 0
LABEL_SMOOTHING_D = 0.4 # Label smoothing в энтропии дискриминатора
LABEL_SMOOTHING_G = 0.4 # Label smoothing в энтропии генератора
DROPOUT_G_ENCODER = 0.35
DROPOUT_G_DECODER = 0.6

# Расширение датасета
USE_FLIP_AUG = True # Добавить поворот зеркальный относительно вертикали
USE_SCALE_INTENSE_AUG = True # Добавить случайное изменение контраста умножением значений изображения на контсанту
USE_SHIFT_INTENSE_AUG = True # Добавить случайное смещение корсета вдоль высоты

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Директории с данными
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
CHECKPOINT_DIR = "checkpoints"

# Число потоков для загрузки картинок в датасеты
NUM_WORKERS = 0

# Логгирование чекпоинтов моделей
CHECKPOINT_DIR = "../../chkpt/"
CHECKPOINT_SAVE_STEP = 100

# Предобученные модели, если продолжаем обучение
PRETRAIN_CHECKPOINT_DISC = "../../discriminator120.pt"
PRETRAIN_CHECKPOINT_GEN = "../../generator120.pt"

# Чекпоинт для скрипта evaluate.py
EVALUATE_CHECKPOINT_GEN = "../../chkpt/generator_1100.pt"

# Общая обработка изображений для обучения и проверки
common_transform = A.Compose(
    [A.Resize(width=IM_SIZE, height=IM_SIZE)],
    additional_targets={"image0": "image"},
)

# Предобработка с расширением датасета для обучения
train_transform = A.Compose(
    [
        A.PadIfNeeded(325, 275),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=10, p=0.3),
        A.OneOf([
            A.RandomCrop(width=IM_SIZE, height=IM_SIZE),
            A.RandomResizedCrop(width=IM_SIZE, height=IM_SIZE, ratio=(1,1), scale=(0.81,1)),
        ], p=1),
    ],
    additional_targets={"image0": "image"},
)

# Нормализация для перевода numpy->Tensor
transform_to_tensor = A.Compose(
    [
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)