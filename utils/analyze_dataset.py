import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision import transforms
from data.dataset import ASLDataset
from utils.vizualization import plot_class_distribution
from utils.vizualization import plot_sample_images

train_dir = "asl_classifier/data/asl_alphabet_train/asl_alphabet_train"
val_dir = "asl_classifier/data/asl_alphabet_val/asl_alphabet_val"
test_dir = "asl_classifier/data/asl_alphabet_test/asl_alphabet_test"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = ASLDataset(train_dir, transform=transform)

plot_class_distribution({
    "train": train_dir,
    "val": val_dir,
    "test": test_dir
})

plot_sample_images(train_dataset, train_dataset.class_to_idx)
