# ASL Classifier (CNN)

CNN-классификатор для изображений американского жестового алфавита (ASL).

## Установка

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

![Training](https://github.com/4pokodav/ASL_classifier/raw/main/plots/training_plot.png)
![Class distribution](https://github.com/4pokodav/ASL_classifier/raw/main/plots/class_distribution.png)
![Sample images](https://github.com/4pokodav/ASL_classifier/raw/main/plots/sample_images.png)
![Confusion matrix](https://github.com/4pokodav/ASL_classifier/raw/main/plots/confusion_matrix.png)
