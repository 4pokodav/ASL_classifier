# ASL Classifier (CNN)

CNN-классификатор для изображений американского жестового алфавита (ASL).

## Установка зависимостей

```
pip install -r requirements.txt
```

## Запуск обучения
```
python training/train.py
```

## Пример предсказания
```
python example.py
```

## Структура проекта
- models/ — модель CNN

- data/ — датасеты и DataLoader

- training/ — скрипты обучения

- utils/ — графики, логгеры и утилиты

## Обзор работы

Модель реализована на PyTorch и представляет собой глубокую сверточную нейронную сеть с пятью сверточными блоками и классификатором. Каждый сверточный блок включает:
- Сверточный слой с ядром 3×3 и паддингом 1
- Нормализацию по батчу
- Функцию активации ReLU
- Субдискретизацию через MaxPool2d
- Dropout для борьбы с переобучением
- В конце используется AdaptiveAvgPool2d для усреднения пространственных признаков и выходной полносвязный слой, преобразующий тензор признаков размером 512 в вектор из 29 классов.
- В обучении использовал оптимизатор Adam и функцию потерь CrossEntropy.

У обученной модели (best_model) точность на тестовой выборке составила - 0.9994

**График обучения:**
![Training](https://github.com/4pokodav/ASL_classifier/raw/main/plots/training_plot.png)

Датасет взял с kaggle (https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

**Разделение данных датасета:**
![Class distribution](https://github.com/4pokodav/ASL_classifier/raw/main/plots/class_distribution.png)

Выборки для каждого экземпляра датасета имеют одинаковый размер, поэтому дисбаланс отсутствует.

**Confusion matrix модели:**
![Confusion matrix](https://github.com/4pokodav/ASL_classifier/raw/main/plots/confusion_matrix.png)

Модель ошиблась только в 1 предсказании для символа "B" и 4 раза для символа "M", в остальном, модель показывает хорошие результаты.
