import os
import shutil
import random
from tqdm import tqdm

def create_val_test_split(
    train_dir='asl_classifier/data/asl_alphabet_train/asl_alphabet_train',
    val_dir='asl_classifier/data/asl_alphabet_val/asl_alphabet_val',
    test_dir='asl_classifier/data/asl_alphabet_test/asl_alphabet_test',
    val_ratio=0.0,
    test_ratio=0.1,
    seed=42
):
    assert val_ratio + test_ratio < 1, "Сумма val_ratio и test_ratio должна быть < 1"
    
    random.seed(seed)

    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_names = sorted(os.listdir(train_dir))

    for class_name in tqdm(class_names, desc="Разделение на train/val/test"):
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        images = [
            f for f in os.listdir(train_class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(images)

        total_count = len(images)
        val_count = int(total_count * val_ratio)
        test_count = int(total_count * test_ratio)

        val_images = images[:val_count]
        test_images = images[val_count:val_count + test_count]

        for img_name in val_images:
            shutil.move(os.path.join(train_class_dir, img_name),
                        os.path.join(val_class_dir, img_name))

        for img_name in test_images:
            shutil.move(os.path.join(train_class_dir, img_name),
                        os.path.join(test_class_dir, img_name))

    print(f"{val_ratio*100:.0f}% данных перемещены в {val_dir}")
    print(f"{test_ratio*100:.0f}% данных перемещены в {test_dir}")

if __name__ == "__main__":
    create_val_test_split()