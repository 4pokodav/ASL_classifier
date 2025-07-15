import torch
from torchvision import transforms
from PIL import Image
from models.cnn_model import CNN

# Загрузка модели
model = CNN(num_classes=29)
model.load_state_dict(torch.load("asl_classifier/models/best_model.pth"))
model.eval()

# Препроцессинг изображения
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Загрузка изображения
img = Image.open("asl_classifier/data/asl_alphabet_test//asl_alphabet_test/del/del_test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)  # Добавляем batch размерности

# Предсказание
with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1).item()

# Сопоставление индекса с символом
idx_to_class = {i: chr(65 + i) for i in range(26)}  # A-Z
idx_to_class[26] = 'del'
idx_to_class[27] = 'nothing'
idx_to_class[28] = 'space'

print(f"Предсказанный класс: {pred} -> Символ: {idx_to_class[pred]}")
