# hot_dog

Сравнение разных методов работы с несбалансированными данными.

Для сравнения методов работы с несбалансированными данными был использован датасет Hot Dog - Not Hot Dog. Для этого случайная часть изображений трейн-выборки из класса "hot_dog" перенесена в тестовую выборку. 20% трейн-выборки используются для валидации. Для обучения использовалась сеть resnet18.

Для сравнения проведены 4 эксперимента:

1. обучение на несбалансированных данных

2. обучение на несбалансированных данных и лосс-функция с весами, полученными на основе количества изображений каждого класса в трейн-выборке

3. обучение с использованием ImbalancedDatasetSampler: https://github.com/ufoym/imbalanced-dataset-sampler﻿

4. обучение с использованием WeightedRandomSampler.

Данные для обучения взяты здесь: https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog

Для скачивания датасета и его предобработки необходимо запустить файл prepare_data.py.
При этом создается папка с валидационной выборкой, часть изображений из трейн выборки переносится в тестовую.

В файле config.py прописаны параметры обучения.

Файл main.py запускает процесс тренировки. По ходу обучения в текущую директорию будут сохранены графики loss, матрица ошибок, распределение классов, ROC-кривая и примеры неправильно классифицированных ихображений.

Эксперименты показали, что сэмплеры улучшают способность сети классифицировать изображения, но не всегда:(

Подробное описание экспериментов здесь: https://wandb.ai/olgatrofimova96/hot_dog/reports/---VmlldzoxNTA1Mzc4?accessToken=9zyzymnm7iwejr7fm72t1giqdfurjc138q8q62zx7w5blu5wiwp5nd7nhkjma2so
