import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
from tqdm import tqdm

from data.dataset import ASLDataset
from models.cnn_model import CNN
from utils.vizualization import save_training_plot
from utils.metrics import evaluate
from utils.logger import get_logger
from training import config
from utils.vizualization import plot_confusion_matrix

logger = get_logger(__name__)
logger.info(f"Используется девайс: {config.DEVICE}")

# Данные
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = ASLDataset(config.TRAIN_DIR, transform)
val_dataset = ASLDataset(config.VAL_DIR, transform)
test_dataset = ASLDataset(config.TEST_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

logger.info(f"Train samples: {len(train_dataset)}")
logger.info(f"Val samples: {len(val_dataset)}")
logger.info(f"Test samples: {len(test_dataset)}")

# Модель
model = CNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LR)

best_val_acc = 0.0
patience_counter = 0
train_losses, val_losses, val_accuracies = [], [], []

# Обучение
logger.info("Запуск обучения")
for epoch in range(1, config.EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Эпоха {epoch:02d}", file=sys.stdout)):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_loss, val_acc = evaluate(model, val_loader, config.DEVICE, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    logger.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), config.MODEL_PATH)
        logger.info("\tСохранена текущая лучшая модель")
    else:
        patience_counter += 1
        logger.info(f"\tБез изменений: Счетчик: {patience_counter}/{config.PATIENCE}")
        if patience_counter >= config.PATIENCE:
            logger.info("\tРанняя остановка")
            break

# Графики
save_training_plot(train_losses, val_losses, val_accuracies, config.PLOT_PATH)
logger.info(f"График обучения сохранен {config.PLOT_PATH}")

# Тестирование 
logger.info("\nТестирование модели")
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Тестирование", file=sys.stdout):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

logger.info(f"Тестовая точность: {evaluate(model, test_loader, config.DEVICE, criterion)[1]:.4f}")
logger.info("\nЛоги обучения:\n" + classification_report(all_labels, all_preds))

# Confusion matrix
class_names = sorted(train_dataset.class_to_idx.keys())
plot_confusion_matrix(all_labels, all_preds, class_names)