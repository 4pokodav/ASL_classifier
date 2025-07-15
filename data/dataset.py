import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple

class ASLDataset(Dataset):
    """
    Dataset для изображений американского жестового алфавита (ASL).

    Структура:
        root_dir/
            A/
                img1.jpg
                ...
            B/
                img1.jpg
                ...
            ...
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples: list[Tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_dataset()

    def _load_dataset(self) -> None:
        class_names = sorted([d for d in os.listdir(self.root_dir)
                              if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for cls_name in class_names:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(cls_folder, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[transforms.Compose, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label