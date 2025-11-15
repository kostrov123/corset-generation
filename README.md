## Pix2Pix модуль
Модуль с реализацией обучения/проверки нейросетевой модели

## Installation
```bash
pip install -r requirements.txt
```

## Dataset
Относительный путь к датасету по-умлочанию:
../data_new


## Train
Запустить projections.py для получения изображений-срезов из stl-моделей

Запустить prepare_embedding.py для получения эмбедингов рентгенов

Настроить pix2pix/config.py если требуется (TRAIN_DIR и VAL_DIR в минимальном варианте)

Запустить pix2pix/train.py скрипт для обучения модели на этих данных


## Tests
Запустить pix2pix/evaluate.py скрипт для получения изображений-предсказаний карт высот

Запустить backprojection.py для получения stl-моделей
