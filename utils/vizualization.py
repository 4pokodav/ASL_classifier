import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import seaborn as sns

def save_training_plot(train_losses, val_losses, val_accuracies, save_path):
    """
    Создает график обучения модели (Функция потерь и точность по эпохам)
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)

def plot_class_distribution(data_dirs: dict, save_path: str = "asl_classifier/plots/class_distribution.png") -> None:
    """
    Рисует распределение классов по датасетам (train, val, test).
    """
    class_counts = {}

    for split_name, dir_path in data_dirs.items():
        counts = {}
        for class_name in sorted(os.listdir(dir_path)):
            class_path = os.path.join(dir_path, class_name)
            if os.path.isdir(class_path):
                num_images = len([
                    f for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                counts[class_name] = num_images
        class_counts[split_name] = counts

    # Рисуем графики
    all_classes = sorted(list(next(iter(class_counts.values())).keys()))
    x = range(len(all_classes))

    plt.figure(figsize=(12, 6))
    width = 0.25
    for i, (split_name, counts) in enumerate(class_counts.items()):
        heights = [counts.get(cls, 0) for cls in all_classes]
        plt.bar([p + width * i for p in x], heights, width=width, label=split_name)

    plt.xticks([p + width for p in x], all_classes, rotation=45)
    plt.ylabel("Количество изображений")
    plt.title("Распределение классов по выборкам")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_sample_images(dataset, class_to_idx, save_path="asl_classifier/plots/sample_images.png", samples_per_class=1):
    """
    Показывает по 1 изображению на каждый класс из датасета.
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_samples = {class_id: [] for class_id in idx_to_class.keys()}

    for img, label in dataset:
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(img)
        if all(len(samples) == samples_per_class for samples in class_samples.values()):
            break

    all_images = [img for class_imgs in class_samples.values() for img in class_imgs]
    all_labels = [idx_to_class[label] for label in class_samples.keys() for _ in range(samples_per_class)]

    grid = make_grid(all_images, nrow=samples_per_class, normalize=True, pad_value=1)
    plt.figure(figsize=(samples_per_class * 2, len(class_samples)))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Примеры изображений по классам')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="asl_classifier/plots/confusion_matrix.png"):
    """
    Строит confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок (Confusion Matrix)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()