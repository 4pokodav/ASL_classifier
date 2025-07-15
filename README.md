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